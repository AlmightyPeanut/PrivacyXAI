import os

import flwr as fl

from .FederatedLearningClient import get_evaluate_fn, generate_client_fn
from .FederatedLearningConfig import FederatedLearningConfig
from .FederatedLearningStrategy import FederatedLearningStrategy
from experiments.model.ModelManager import ModelManager
from experiments.dataset.DatasetManager import DATASET_MANAGER


class FederatedLearningManager:
    def __init__(self,
                 config: FederatedLearningConfig = FederatedLearningConfig()):
        self.config = config

        self.federated_learning_strategy = None
        self.client_fn_callback = None
        self.current_dataset = None
        self.model_manager = None

    def prepare_simulation(self, dataset_name: str) -> None:
        self.current_dataset = dataset_name
        test_data = DATASET_MANAGER.get_test_data(self.current_dataset)

        self.federated_learning_strategy = FederatedLearningStrategy(
            fraction_fit=self.config.fraction_fit,
            fraction_evaluate=self.config.fraction_evaluate,
            min_available_clients=self.config.number_of_clients,
            evaluate_fn=get_evaluate_fn(test_data)
        )

        train_data_loaders, validation_data_loaders = DATASET_MANAGER.get_federated_learning_data_loaders(
            self.current_dataset, self.config.number_of_clients, self.config.validation_split)

        self.client_fn_callback = generate_client_fn(train_data_loaders, validation_data_loaders,
                                                     self.config.privatise_models)
        self.model_manager = ModelManager()

    def start_simulation(self):
        history = fl.simulation.start_simulation(
            client_fn=self.client_fn_callback,
            num_clients=self.config.number_of_clients,
            config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
            strategy=self.federated_learning_strategy,
        )
        return history.metrics_centralized

    def evaluate_server_model(self) -> dict[str, dict[str, float]]:
        self.model_manager.set_parameters_of_models(self.federated_learning_strategy.get_parameters_of_models())
        evaluation_scores = self.model_manager.evaluate_target_models(
            DATASET_MANAGER.get_test_data(self.current_dataset))
        return evaluation_scores

    def save_server_model(self, model_folder_path: os.PathLike) -> None:
        fl_parameters = {
            "fl": True,
            "fl_clients": self.config.number_of_clients,
            "fl_rounds": self.config.num_rounds,
            "privatised": self.config.privatise_models,
        }
        self.model_manager.save_models(model_folder_path, fl_parameters)

    def get_server_model_parameters(self):
        return self.federated_learning_strategy.get_parameters_of_models()


FEDERATED_LEARNING_MANAGER = FederatedLearningManager()
