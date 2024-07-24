import numpy as np
import torch.utils.data
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, random_split

from .BreastCancerDataset import BreastCancerDataset
from .IrisDataset import IrisDataset
from .MIMICDataset import MIMICDataset
from .DatasetConfig import DatasetConfig, SUPPORTED_DATASETS
from experiments.utils.Singleton import Singleton
from .MIMICExtract import MIMICExtractDataset
from .NormalisedDataset import NormalisedDataset

torch.manual_seed(42)


def collate_fn(x):
    return x


class DatasetManager(metaclass=Singleton):
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        self.config = config
        self.collate_fn = collate_fn
        self.datasets = dict()

        self.federated_learning_dataloaders = {}
        self._load_datasets()

    def _load_datasets(self) -> None:
        if self.datasets:
            return

        for dataset_name in self.config.datasets.intersection(SUPPORTED_DATASETS):
            if dataset_name == 'Iris':
                dataset = IrisDataset()
            elif dataset_name == 'BreastCancer':
                dataset = BreastCancerDataset()
            elif dataset_name == 'MIMICExtract':
                dataset = MIMICExtractDataset()
            elif dataset_name == 'MIMIC':
                dataset = MIMICDataset()
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not supported")

            self.datasets[dataset_name] = dict()

            dataset_labels = dataset.classes.to_numpy().astype(np.int64).squeeze(0)
            k_fold = StratifiedKFold(n_splits=self.config.kfolds, shuffle=True, random_state=42)
            folds_element_ids = [fold_ids for _, fold_ids in
                                 k_fold.split(np.zeros(len(dataset)),
                                              dataset_labels)]

            for fold_index, (train_ids, test_ids, mia_non_member_ids) in enumerate(
                    self.prepare_train_test_mia_fold_ids(folds_element_ids)
            ):
                train_data_features = dataset.__getitems__(train_ids)["features"]
                train_std = np.std(train_data_features, axis=0)
                train_mean = np.mean(train_data_features, axis=0)
                del train_data_features

                normalised_dataset = NormalisedDataset(dataset, train_std, train_mean)

                self.datasets[dataset_name][fold_index] = dict()
                self.datasets[dataset_name][fold_index]["train"] = DataLoader(
                    torch.utils.data.Subset(normalised_dataset, train_ids),
                    batch_size=self.config.batch_size if self.config.batch_size > 0 else len(train_ids),
                    shuffle=True,
                    collate_fn=self.collate_fn
                )
                self.datasets[dataset_name][fold_index]["test"] = DataLoader(
                    torch.utils.data.Subset(normalised_dataset, test_ids),
                    batch_size=len(test_ids),
                    shuffle=True,
                    collate_fn=self.collate_fn
                )
                # This represents either held back data or newly acquired data by a malicious client
                self.datasets[dataset_name][fold_index]["mia"] = DataLoader(
                    torch.utils.data.Subset(normalised_dataset, mia_non_member_ids),
                    batch_size=len(mia_non_member_ids),
                    shuffle=True,
                    collate_fn=self.collate_fn
                )

    def get_data_folds(self, dataset_name: str) -> (int, DataLoader, DataLoader):
        if not self.datasets[dataset_name]:
            raise ValueError(f"Dataset {dataset_name} not available")

        if self.datasets[dataset_name] is None:
            raise RuntimeError("Datasets have to be loaded first.")

        for fold_index, data in self.datasets[dataset_name].items():
            yield fold_index, (data["train"], data["test"])

    def get_mia_data_folds(self, dataset_name: str, use_federated_data_only: bool = False,
                           number_of_clients: int = 0) -> (int, np.array, DataLoader, DataLoader):
        if not self.datasets[dataset_name]:
            raise ValueError(f"Dataset {dataset_name} not available")

        if self.datasets[dataset_name] is None:
            raise RuntimeError("Datasets have to be loaded first.")

        for fold_index, data in self.datasets[dataset_name].items():
            train_data = data["train"]

            if use_federated_data_only:
                if number_of_clients <= 0:
                    raise ValueError("Number of clients must be greater than zero if federated data should be used!")
                train_data = self.split_data_for_federated_learning(train_data, number_of_clients)[0]

            train_data_result = {
                "features": [],
                "classes": [],
            }

            for train_data_dict in train_data:
                train_data_result["features"].append(train_data_dict["features"])
                train_data_result["classes"].append(train_data_dict["classes"])

            train_data_result["features"] = np.concatenate(train_data_result["features"], axis=0)
            train_data_result["classes"] = np.concatenate(train_data_result["classes"], axis=0)

            yield fold_index, (train_data_result, data["test"], data["mia"])

    def split_data_for_federated_learning(self, data: DataLoader, number_of_clients: int) -> list[DataLoader]:
        train_data = data.dataset

        partition_length = len(train_data) // number_of_clients
        remaining_items = len(train_data) - number_of_clients * partition_length
        partition_sizes = ([partition_length + 1] * remaining_items
                           + [partition_length] * (number_of_clients - remaining_items))

        data_partitions = random_split(train_data, partition_sizes)
        data_loaders = [
            DataLoader(data_partition,
                       batch_size=self.config.batch_size if self.config.batch_size > 0 else len(data_partition),
                       shuffle=True,
                       collate_fn=self.collate_fn)
            for data_partition in data_partitions
        ]

        print("Number of local training samples: {}".format(
            len(data_loaders[0].dataset)))

        return data_loaders

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
            return self.datasets[dataset_name]["val"]

        return self.federated_learning_dataloaders[dataset_name]["val"][0]

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

        return DataLoader(data,
                          batch_size=self.config.batch_size if self.config.batch_size > 0
                          else len(data), shuffle=True,
                          collate_fn=self.collate_fn)

    def get_number_of_features(self, dataset_name: str) -> int:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        return self.datasets[dataset_name][0]['train'].dataset.dataset.get_number_of_features()

    def get_number_of_classes(self, dataset_name: str) -> int:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} not available")

        return self.datasets[dataset_name][0]['train'].dataset.dataset.get_number_of_classes()

    @staticmethod
    def prepare_train_test_mia_fold_ids(folds_element_ids: list[np.array]):
        for fold_index, fold_element_ids in enumerate(folds_element_ids):
            train_ids = [ids for index, ids in enumerate(folds_element_ids) if index != fold_index]
            mia_ids = train_ids[-1]
            train_ids = np.concatenate(train_ids[:-1])
            yield train_ids, fold_element_ids, mia_ids


DATASET_MANAGER = DatasetManager()
