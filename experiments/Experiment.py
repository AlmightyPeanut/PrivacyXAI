import json

from torch.utils.data import DataLoader

from experiments.dataset.DatasetManager import DATASET_MANAGER
from experiments.model.ModelManager import ModelManager
from experiments.federated_learning.FederatedLearningManager import FederatedLearningManager
from experiments.utils import MODEL_CHECKPOINTS_PATH, RESULTS_PATH, PRINT_WIDTH
from experiments.xai.XAIManager import XAIManager
from experiments.mia.MIAManager import MIA_MANAGER


class Experiment:
    def __init__(self, number_of_clients: list[int], epsilons: list[float]):
        self.number_of_clients = number_of_clients
        self.epsilons = epsilons

    def run_model_training(self, federated_learning: bool = True, centralised_model: bool = True):
        for dataset_name in DATASET_MANAGER.datasets:
            if federated_learning:
                self._run_federated_learning(dataset_name)
            if centralised_model:
                self._run_centralised_model(dataset_name)

    def _run_centralised_model(self, dataset_name: str):
        for fold_index, (train_data, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
            self._run_centralised_training(dataset_name, train_data, test_data, fold_index,
                                           use_differential_privacy=False)

            for epsilon in self.epsilons:
                self._run_centralised_training(dataset_name, train_data, test_data, fold_index,
                                               use_differential_privacy=True, epsilon=epsilon)

    @staticmethod
    def _run_centralised_training(dataset_name: str, train_data: DataLoader, test_data: DataLoader, fold_index: int,
                                  use_differential_privacy: bool, epsilon: float = .0):
        model_manager = ModelManager(DATASET_MANAGER.get_number_of_features(dataset_name),
                                     DATASET_MANAGER.get_number_of_classes(dataset_name))
        if use_differential_privacy:
            model_manager.privatise_models_and_data(train_data, epsilon=epsilon)

        print(f"Training target model with {dataset_name} training data. Fold {fold_index}")
        model_manager.train_target_models(train_data)
        model_manager.save_models(MODEL_CHECKPOINTS_PATH / 'non_fl_model',
                                  {'privatised': use_differential_privacy, 'fl': False, 'fold': fold_index,
                                   'epsilon': epsilon})

        print(f"Testing model with {dataset_name} test data. Fold {fold_index}")
        model_manager.evaluate_target_models(test_data, fold_index, {
            'privatised': use_differential_privacy,
            'epsilon': epsilon,
            'fl': False,
        })

    def _run_federated_learning(self, dataset_name: str):
        for number_of_clients in self.number_of_clients:
            fl_manager = FederatedLearningManager(privatise_models=False, number_of_clients=number_of_clients)
            fl_manager.start_simulation(dataset_name)

            for epsilon in self.epsilons:
                fl_manager = FederatedLearningManager(privatise_models=True, number_of_clients=number_of_clients,
                                                      epsilon=epsilon)
                fl_manager.start_simulation(dataset_name)

    def run_xai_evaluation(self, use_federated_model=True, use_centralised_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" XAI evaluation on {dataset_name} ".center(PRINT_WIDTH, '#'))

            model_paths = []
            if use_federated_model:
                model_path = MODEL_CHECKPOINTS_PATH / 'fl_server_model'
                for model_file in model_path.iterdir():
                    if not str(model_file).endswith('.pth'):
                        continue
                    model_paths.append(model_path / model_file)

            if use_centralised_model:
                model_path = MODEL_CHECKPOINTS_PATH / 'non_fl_model'
                for model_file in model_path.iterdir():
                    if not str(model_file).endswith('.pth'):
                        continue
                    model_paths.append(model_path / model_file)

            for fold_index, (_, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
                fold_models = [p for p in model_paths if f"fold={fold_index}" in str(p)]
                XAIManager().evaluate_explanations(test_data,
                                                   DATASET_MANAGER.get_number_of_features(dataset_name),
                                                   DATASET_MANAGER.get_number_of_classes(dataset_name),
                                                   fold_models)

    @staticmethod
    def run_mia(use_federated_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" MIA on {dataset_name} ".center(PRINT_WIDTH, '#'))

            if use_federated_model:
                model_path = MODEL_CHECKPOINTS_PATH / 'fl_server_model'
            else:
                model_path = MODEL_CHECKPOINTS_PATH / 'non_fl_model'

            mia_scores = MIA_MANAGER.run_membership_inference_attack(dataset_name, model_path, use_federated_model)
            json_file_name = 'fl_model_mia_metrics.json'
            if not use_federated_model:
                json_file_name = 'non_' + json_file_name
            with open(RESULTS_PATH / json_file_name, 'w') as f:
                json.dump(mia_scores, f)

            for model_name, mia_metric_scores in mia_scores.items():
                print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
                for mia_metric_name, mia_metric_score in mia_metric_scores.items():
                    print(f"{mia_metric_name}: {mia_metric_score}, ", end='')
                print()
