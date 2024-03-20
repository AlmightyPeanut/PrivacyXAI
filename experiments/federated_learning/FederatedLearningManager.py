import os

import flwr as fl
from torch.utils.data import DataLoader, random_split

from .FederatedLearningClient import get_evaluate_fn, generate_client_fn
from .FederatedLearningConfig import FederatedLearningConfig
from .FederatedLearningStrategy import FederatedLearningStrategy
from ..model.ModelConfig import ModelConfig
from ..model.ModelManager import ModelManager


class FederatedLearningManager:
    def __init__(self,
                 data_loader: DataLoader, test_data: DataLoader,
                 config: FederatedLearningConfig = FederatedLearningConfig()):
        self.config = config
        self.federated_learning_strategy = FederatedLearningStrategy(
            fraction_fit=self.config.fraction_fit,
            fraction_evaluate=self.config.fraction_evaluate,
            min_available_clients=self.config.number_of_clients,
            evaluate_fn=get_evaluate_fn(test_data)
        )

        train_data_loaders, validation_data_loaders = self._prepare_dataset_for_federated_learning(
            data_loader, number_of_partitions=self.config.number_of_clients)

        self.client_fn_callback = generate_client_fn(train_data_loaders, validation_data_loaders,
                                                     self.config.privatise_models)

    def start_simulation(self):
        history = fl.simulation.start_simulation(
            client_fn=self.client_fn_callback,
            num_clients=self.config.number_of_clients,
            config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
            strategy=self.federated_learning_strategy,
        )
        return history.metrics_centralized

    def evaluate_server_model(self, eval_data: DataLoader) -> dict[str, dict[str, float]]:
        model_manager = ModelManager()
        model_manager.set_parameters_of_models(self.federated_learning_strategy.get_parameters_of_models())
        evaluation_scores = model_manager.evaluate_target_models(eval_data)
        return evaluation_scores

    def save_server_model(self, path: os.PathLike) -> None:
        self.federated_learning_strategy.save_models(path)

    def get_server_model_parameters(self):
        return self.federated_learning_strategy.get_parameters_of_models()

    def _prepare_dataset_for_federated_learning(self, data_loader: DataLoader, number_of_partitions: int = 5) -> (
            list[DataLoader], list[DataLoader]):
        partition_length = len(data_loader.dataset) // number_of_partitions
        remaining_items = len(data_loader.dataset) - number_of_partitions * partition_length
        partition_sizes = [partition_length + 1] * remaining_items + [partition_length] * (
                number_of_partitions - remaining_items)

        data_partitions = random_split(data_loader.dataset, partition_sizes)

        train_data_loaders = []
        validation_data_loaders = []
        for data_partition in data_partitions:
            n_total_samples = len(data_partition)
            n_validation_samples = max(int(self.config.validation_ratio * n_total_samples), 10)
            n_training_samples = n_total_samples - n_validation_samples

            if n_training_samples <= 0:
                raise ValueError(
                    f"Number of training samples should be greater than 0. Try adjusting the validation split.")

            local_train_set, local_validation_dataset = random_split(
                data_partition,
                [n_training_samples, n_validation_samples])

            train_data_loaders.append(DataLoader(local_train_set, batch_size=self.config.batch_size, shuffle=True,
                                                 collate_fn=lambda x: x))
            validation_data_loaders.append(DataLoader(local_validation_dataset, batch_size=n_validation_samples,
                                                      collate_fn=lambda x: x))

        print("Number of local training samples: {}".format(len(train_data_loaders[0].dataset)))
        print("Number of local validation samples: {}".format(len(validation_data_loaders[0].dataset)))

        return train_data_loaders, validation_data_loaders
