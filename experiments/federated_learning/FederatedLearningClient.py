from typing import Union

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from experiments.model.ModelManager import ModelManager

# python 3.9 compatability
scalar = Union[bool, bytes, float, int, str]


class FederatedLearningClient(fl.client.NumPyClient):
    def __init__(self, train_data: DataLoader, number_of_features: int, number_of_classes: int, privatise_models: bool,
                 epsilon: float = .0):
        super().__init__()
        self.train_data = train_data

        self.train_data_size = len(train_data)

        self.model_manager = ModelManager(number_of_features, number_of_classes)
        if privatise_models:
            self.model_manager.privatise_models_and_data(train_data, epsilon)

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
        evaluation_scores = self.model_manager.evaluate_target_models(self.train_data)

        return .0, self.train_data_size, evaluation_scores
