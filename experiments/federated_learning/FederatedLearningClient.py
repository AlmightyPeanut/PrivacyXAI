from typing import Union

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from experiments.model.ModelConfig import ModelConfig
from experiments.model.ModelManager import ModelManager

# python 3.9 compatability
scalar = Union[bool, bytes, float, int, str]


class FederatedLearningClient(fl.client.NumPyClient):
    def __init__(self, train_data: DataLoader, number_of_features: int, number_of_classes: int, privatise_models: bool,
                 epsilon: float = .0, local_training_rounds: int = 1):
        super().__init__()
        self.train_data = train_data

        self.train_data_size = len(train_data)

        model_config = ModelConfig()
        model_config.number_of_epochs = 1
        self.model_manager = ModelManager(number_of_features, number_of_classes, config=model_config)
        if privatise_models:
            self.train_data = self.model_manager.privatise_models_and_data(train_data, epsilon)

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
    ) -> tuple[float, int, dict[str, float]]:
        self.model_manager.set_parameters_of_models(parameters_of_models)
        evaluation_scores = self.model_manager.evaluate_target_models(self.train_data, fold_index=0,
                                                                      model_parameters={}, save_results=False)

        return .0, self.train_data_size, evaluation_scores["NN"]
