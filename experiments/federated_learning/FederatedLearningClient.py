from typing import Callable

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from experiments.model.ModelManager import ModelManager

scalar = bool | bytes | float | int | str


class FederatedLearningClient(fl.client.NumPyClient):
    def __init__(self, train_data: DataLoader, validation_data: DataLoader, privatise_models: bool):
        super().__init__()
        self.train_data = train_data
        self.validation_data = validation_data

        self.train_data_size = len(train_data)
        self.validation_data_size = len(validation_data)

        self.model_manager = ModelManager()
        if privatise_models:
            self.model_manager.privatise_models_and_data(train_data)

    def get_parameters(self, config: dict[str, scalar]) -> list[np.array]:
        return self.model_manager.get_parameters_of_models()

    def fit(
            self, parameters_of_models: list[np.array], config: dict[str, scalar]
    ) -> (dict[str, list[np.array]], int, dict[str, scalar]):
        self.model_manager.set_parameters_of_models(parameters_of_models)
        self.model_manager.train_target_models(self.train_data)

        return self.get_parameters({}), self.train_data_size, {}

    def evaluate(
            self, parameters_of_models: list[np.array], config: dict[str, scalar]
    ) -> tuple[float, int, dict[str, dict[str, float]]]:
        self.model_manager.set_parameters_of_models(parameters_of_models)
        evaluation_scores = self.model_manager.evaluate_target_models(self.validation_data)

        return .0, self.validation_data_size, evaluation_scores


def generate_client_fn(train_data_loaders: list[DataLoader], validation_data_loaders: list[DataLoader],
                       privatise_models: bool = True) -> Callable:
    def generate_client(client_id: str) -> FederatedLearningClient:
        client_id = int(client_id)
        return FederatedLearningClient(
            train_data=train_data_loaders[client_id],
            validation_data=validation_data_loaders[client_id],
            privatise_models=privatise_models
        )

    return generate_client


def get_evaluate_fn(test_data_loader: DataLoader) -> Callable:
    """ Server evaluation function. Does not return a loss value."""
    def evaluate(server_round: int, parameters_of_models: list[np.array],
                 config: dict[str, scalar]) -> (float, dict[dict[str, scalar]]):
        model_manager = ModelManager()
        model_manager.set_parameters_of_models(parameters_of_models)
        evaluation_scores = model_manager.evaluate_target_models(test_data_loader)

        return .0, evaluation_scores

    return evaluate
