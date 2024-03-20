from pathlib import Path

from torch.utils.data import DataLoader

from dataset.DatasetManager import DatasetManager
from experiments.federated_learning.FederatedLearningManager import FederatedLearningManager
from experiments.model.ModelManager import ModelManager


class Experiment:
    def __init__(self):
        self.dataset_manager = DatasetManager()

    def run_model_training(self, federated_learning: bool = True, privatise_non_fl_models: bool = True):
        for dataset_name, (train_data, test_data) in self.dataset_manager.load_datasets().items():
            if federated_learning:
                self._run_federated_learning(dataset_name, train_data, test_data)
            else:
                self._run(dataset_name, train_data, test_data, privatise_non_fl_models)

    def _run(self, dataset_name: str, train_data: DataLoader, test_data: DataLoader, privatise_data):
        model_manager = ModelManager()

        if privatise_data:
            model_manager.privatise_models_and_data(train_data)

        print(f"Training target model with {dataset_name} training data")
        model_manager.train_target_models(train_data)
        model_manager.save_models(Path(__file__) / 'model_checkpoints' / 'non_fl_model')

        print(f"Testing model with {dataset_name} test data")
        metrics = model_manager.evaluate_target_models(test_data)

        print(f" Training results for {dataset_name} ".center(40, '#'))
        for model_name, metric_scores in metrics.items():
            print(f" Model name: {model_name} ".center(40, '_'))
            for metric_name, metric_score in metric_scores.items():
                print(f"{metric_name}: {metric_score}")

    def _run_federated_learning(self, dataset_name: str, train_data: DataLoader, test_data: DataLoader):
        federated_learning_manager = FederatedLearningManager(train_data, test_data)

        results = federated_learning_manager.start_simulation()
        server_model_results = federated_learning_manager.evaluate_server_model(test_data)
        federated_learning_manager.save_server_model(Path(__file__).parent / 'model_checkpoints' / 'fl_server_model')

        print(f" FL training results for {dataset_name} ".center(40, '#'))

        print(f"Client training results".center(40, '_'))
        for model_name, iteration_results in results.items():
            print(f" Model name: {model_name} ".center(40, '-'))
            for iteration, metric_scores in iteration_results:
                print(f"Iteration {iteration}: ", end='')
                for metric_name, metric_score in metric_scores.items():
                    print(f"{metric_name}: {metric_score}, ", end='')
                print()
            print()
        print()

        print(f"Server training results".center(40, '_'))
        for model_name, metric_scores in server_model_results.items():
            print(f" Model name: {model_name} ".center(40, '-'))
            for metric_name, metric_score in metric_scores.items():
                print(f"{metric_name}: {metric_score}, ", end='')
            print()
            print()
