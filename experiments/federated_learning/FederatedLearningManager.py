import json
from typing import Callable

import flwr as fl
import numpy as np
from torch.utils.data import DataLoader

from .FederatedLearningClient import scalar, FederatedLearningClient
from .FederatedLearningConfig import FederatedLearningConfig
from .FederatedLearningStrategy import FederatedLearningStrategy
from experiments.model.ModelManager import ModelManager
from experiments.dataset.DatasetManager import DATASET_MANAGER
from ..utils import MODEL_CHECKPOINTS_PATH, RESULTS_PATH, PRINT_WIDTH


class FederatedLearningManager:
    def __init__(self,
                 privatise_models: bool,
                 number_of_clients: int,
                 epsilon: float = .0,
                 config: FederatedLearningConfig = FederatedLearningConfig()):
        self.config = config
        self.use_differential_privacy = privatise_models
        self.number_of_clients = number_of_clients
        self.epsilon = epsilon

    def start_simulation(self, dataset_name: str) -> None:
        fold_results = {}
        number_of_features = DATASET_MANAGER.get_number_of_features(dataset_name)
        number_of_classes = DATASET_MANAGER.get_number_of_classes(dataset_name)

        for fold_index, (train_data, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
            federated_learning_strategy = FederatedLearningStrategy(
                fraction_fit=self.config.fraction_fit,
                fraction_evaluate=self.config.fraction_evaluate,
                min_available_clients=self.number_of_clients,
                evaluate_fn=self.get_evaluate_fn(test_data, number_of_features, number_of_classes),
            )

            train_data_loaders = DATASET_MANAGER.split_data_for_federated_learning(train_data,
                                                                                   self.number_of_clients)
            client_fn_callback = self.generate_client_fn(train_data_loaders, number_of_features, number_of_classes)

            history = fl.simulation.start_simulation(
                client_fn=client_fn_callback,
                num_clients=self.number_of_clients,
                config=fl.server.ServerConfig(num_rounds=self.config.num_rounds),
                strategy=federated_learning_strategy,
            )

            fold_result = history.metrics_centralized
            file_name = f'fl_model_metrics_privatised={self.use_differential_privacy}_fl=True_fold={fold_index}.json'
            with open(RESULTS_PATH / file_name, 'w') as f:
                json.dump(fold_results, f)
            fold_results[fold_index] = fold_result

            print(f" FL training results for {dataset_name} ".center(PRINT_WIDTH, '#'))
            print(f"Client training results".center(PRINT_WIDTH, '_'))
            for model_name, iteration_results in fold_result.items():
                print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
                for iteration, metric_scores in iteration_results:
                    print(f"Iteration {iteration}: ", end='')
                    for metric_name, metric_score in metric_scores.items():
                        print(f"{metric_name}: {metric_score}, ", end='')
                    print()
                print()
            print()

            # save server model
            model_manager = ModelManager(number_of_features, number_of_classes)
            model_manager.set_parameters_of_models(federated_learning_strategy.get_parameters_of_models())
            fl_parameters = {
                "fl": True,
                "fl_clients": self.number_of_clients,
                "fl_rounds": self.config.num_rounds,
                "privatised": self.use_differential_privacy,
                "epsilon": self.epsilon,
                "fold": fold_index,
            }
            model_manager.save_models(MODEL_CHECKPOINTS_PATH / 'fl_server_model', fl_parameters)

    def generate_client_fn(self, train_data_loaders: list[DataLoader], number_of_features: int,
                           number_of_classes: int) -> Callable:
        def generate_client(client_id: str) -> FederatedLearningClient:
            client_id = int(client_id)
            return FederatedLearningClient(
                train_data_loaders[client_id],
                number_of_features,
                number_of_classes,
                self.use_differential_privacy,
                self.epsilon
            )

        return generate_client

    @staticmethod
    def get_evaluate_fn(test_data_loader: DataLoader, number_of_features: int, number_of_classes: int) -> Callable:
        """ Server evaluation function. Does not return a loss value."""

        def evaluate(server_round: int, parameters_of_models: list[np.array],
                     config: dict[str, scalar]) -> (float, dict[dict[str, scalar]]):
            model_manager = ModelManager(number_of_features, number_of_classes)
            model_manager.set_parameters_of_models(parameters_of_models)
            evaluation_scores = model_manager.evaluate_target_models(test_data_loader)

            return .0, evaluation_scores

        return evaluate
