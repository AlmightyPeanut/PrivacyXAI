import torch.utils.data
from torch.utils.data import DataLoader, random_split

from .MIMICDataset import MIMICDataset
from .DatasetConfig import DatasetConfig, SUPPORTED_DATASETS
from experiments.utils.Singleton import Singleton


class DatasetManager(metaclass=Singleton):
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        self.config = config
        self.collate_fn = lambda x: x
        self.datasets = dict()

        self.federated_learning_dataloaders = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        # TODO: always split the same way
        if self.datasets:
            return

        for dataset_name in self.config.datasets.union(SUPPORTED_DATASETS):
            if dataset_name == 'MIMIC':
                dataset = MIMICDataset()
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not supported")

            train_data, test_data = random_split(
                dataset,
                [len(dataset) - self.config.test_split, self.config.test_split]
            )

            dataset.add_train_test_indicator(test_data.indices)

            self.datasets[dataset_name] = {
                "train": DataLoader(train_data, batch_size=self.config.batch_size, collate_fn=self.collate_fn),
                "test": DataLoader(test_data, batch_size=len(test_data), collate_fn=self.collate_fn)
            }

            # split for federated learning
            partition_length = len(train_data) // self.config.number_of_partitions
            remaining_items = len(train_data) - self.config.number_of_partitions * partition_length
            partition_sizes = [partition_length + 1] * remaining_items + [partition_length] * (
                    self.config.number_of_partitions - remaining_items)

            data_partitions = random_split(train_data, partition_sizes)

            if dataset_name not in self.federated_learning_dataloaders:
                self.federated_learning_dataloaders[dataset_name] = {
                    "train": [],
                    "val": []
                }

            for data_partition in data_partitions:
                n_total_samples = len(data_partition)
                n_validation_samples = max(int(self.config.fl_validation_ratio * n_total_samples), 10)
                n_training_samples = n_total_samples - n_validation_samples

                if n_training_samples <= 0:
                    raise ValueError(
                        f"Number of training samples should be greater than 0. Try adjusting the validation ratio.")

                local_train_set, local_validation_dataset = random_split(
                    data_partition,
                    [n_training_samples, n_validation_samples])

                self.federated_learning_dataloaders[dataset_name]["train"].append(
                    DataLoader(local_train_set, batch_size=self.config.batch_size, shuffle=True,
                               collate_fn=lambda x: x))
                self.federated_learning_dataloaders[dataset_name]["val"].append(
                    DataLoader(local_validation_dataset, batch_size=n_validation_samples, collate_fn=lambda x: x))

            print("Number of local training samples: {}".format(
                len(self.federated_learning_dataloaders[dataset_name]["train"][0].dataset)))
            print("Number of local validation samples: {}".format(
                len(self.federated_learning_dataloaders[dataset_name]["val"][0].dataset)))

    def get_train_data(self, dataset_name: str) -> DataLoader:
        if not self.datasets[dataset_name]:
            raise ValueError(f"Dataset {dataset_name} not available")

        if self.datasets[dataset_name] is not None:
            return self.datasets[dataset_name]["train"]

        raise RuntimeError("Datasets have to be loaded first.")

    def get_test_data(self, dataset_name: str) -> DataLoader:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        if self.datasets[dataset_name] is not None:
            return self.datasets[dataset_name]["test"]

        raise RuntimeError("Datasets have to be loaded first.")

    def get_federated_learning_data_loaders(self, dataset_name: str) -> (list[DataLoader], list[DataLoader]):
        if dataset_name not in self.federated_learning_dataloaders:
            raise ValueError(f"Dataset {dataset_name} not available")

        return (self.federated_learning_dataloaders[dataset_name]["train"],
                self.federated_learning_dataloaders[dataset_name]["val"])

    def get_attacker_train_data(self, dataset_name: str, use_federated_learning_data: bool) -> DataLoader:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        if use_federated_learning_data:
            if dataset_name not in self.federated_learning_dataloaders:
                raise ValueError(f"Dataset {dataset_name} not available")
            return self.datasets[dataset_name]["train"]

        return self.federated_learning_dataloaders[dataset_name]["train"][0]

    def get_attacker_test_data(self, dataset_name: str, use_federated_learning_data: bool) -> DataLoader:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        if use_federated_learning_data:
            if dataset_name not in self.federated_learning_dataloaders:
                raise ValueError(f"Dataset {dataset_name} not available")
            return self.datasets[dataset_name]["test"]

        return self.federated_learning_dataloaders[dataset_name]["test"][0]

    def get_records_to_attack(self, dataset_name: str, use_federated_learning_data: bool) -> DataLoader:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        data = self.datasets[dataset_name]['train'].dataset
        if use_federated_learning_data:
            if dataset_name not in self.federated_learning_dataloaders:
                raise ValueError(f"Dataset {dataset_name} not available")

            all_other_indices = []
            for data_loader in self.federated_learning_dataloaders[dataset_name]["train"][1:]:
                all_other_indices.extend(data_loader.dataset.indices)
            for data_loader in self.federated_learning_dataloaders[dataset_name]["val"][1:]:
                all_other_indices.extend(data_loader.dataset.indices)
            # TODO: Why are there duplicates in the fl data sets?
            data = torch.utils.data.Subset(data.dataset, list(set(all_other_indices)))

        return DataLoader(data, batch_size=self.config.batch_size, shuffle=True,
                          collate_fn=lambda x: x)


DATASET_MANAGER = DatasetManager()
