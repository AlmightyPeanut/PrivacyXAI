from experiments.dataset.DatasetManager import DATASET_MANAGER
from experiments.model.ModelManager import ModelManager
from experiments.federated_learning.FederatedLearningManager import FEDERATED_LEARNING_MANAGER
from experiments.utils import MODEL_CHECKPOINTS_PATH
from experiments.xai.XAIManager import XAI_MANAGER
from experiments.mia.MIAManager import MIA_MANAGER

PRINT_WIDTH = 40


class Experiment:
    def __init__(self):
        pass

    def run_model_training(self, federated_learning: bool = True, privatise_non_fl_models: bool = True):
        for dataset_name in DATASET_MANAGER.datasets:
            if federated_learning:
                self._run_federated_learning(dataset_name)
            else:
                self._run(dataset_name, privatise_non_fl_models)

    def _run(self, dataset_name: str, privatise_data):
        train_data = DATASET_MANAGER.get_train_data(dataset_name)

        model_manager = ModelManager()
        if privatise_data:
            model_manager.privatise_models_and_data(train_data)

        print(f"Training target model with {dataset_name} training data")
        model_manager.train_target_models(train_data)
        model_manager.save_models(MODEL_CHECKPOINTS_PATH / 'non_fl_model')

        print(f"Testing model with {dataset_name} test data")
        metrics = model_manager.evaluate_target_models(DATASET_MANAGER.get_test_data(dataset_name))

        print(f" Training results for {dataset_name} ".center(PRINT_WIDTH, '#'))
        for model_name, metric_scores in metrics.items():
            print(f" Model name: {model_name} ".center(PRINT_WIDTH, '_'))
            for metric_name, metric_score in metric_scores.items():
                print(f"{metric_name}: {metric_score}")

    def _run_federated_learning(self, dataset_name: str):
        FEDERATED_LEARNING_MANAGER.prepare_simulation(dataset_name)

        results = FEDERATED_LEARNING_MANAGER.start_simulation()
        server_model_results = FEDERATED_LEARNING_MANAGER.evaluate_server_model()
        FEDERATED_LEARNING_MANAGER.save_server_model(MODEL_CHECKPOINTS_PATH / 'fl_server_model')

        print(f" FL training results for {dataset_name} ".center(PRINT_WIDTH, '#'))

        print(f"Client training results".center(PRINT_WIDTH, '_'))
        for model_name, iteration_results in results.items():
            print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
            for iteration, metric_scores in iteration_results:
                print(f"Iteration {iteration}: ", end='')
                for metric_name, metric_score in metric_scores.items():
                    print(f"{metric_name}: {metric_score}, ", end='')
                print()
            print()
        print()

        print(f"Server training results".center(PRINT_WIDTH, '_'))
        for model_name, metric_scores in server_model_results.items():
            print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
            for metric_name, metric_score in metric_scores.items():
                print(f"{metric_name}: {metric_score}, ", end='')
            print()
            print()

    def run_xai_evaluation(self, use_federated_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" XAI evaluation on {dataset_name} ".center(PRINT_WIDTH, '#'))

            if use_federated_model:
                model_path = MODEL_CHECKPOINTS_PATH / 'fl_server_model'
            else:
                model_path = MODEL_CHECKPOINTS_PATH / 'non_fl_model'

            xai_scores = XAI_MANAGER.evaluate_explanations(DATASET_MANAGER.get_test_data(dataset_name), model_path)

            for model_name, metric_scores in xai_scores.items():
                print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
                for metric_name, xai_metric_scores in metric_scores.items():
                    print(f"{metric_name}".center(PRINT_WIDTH, '_'))
                    for xai_metric_name, xai_metric_score in xai_metric_scores.items():
                        print(f"{xai_metric_name}: {xai_metric_score}, ", end='')
                    print()
                print()
                print()

    def run_mia(self, use_federated_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" MIA on {dataset_name} ".center(PRINT_WIDTH, '#'))

            if use_federated_model:
                model_path = MODEL_CHECKPOINTS_PATH / 'fl_server_model'
            else:
                model_path = MODEL_CHECKPOINTS_PATH / 'non_fl_model'

            mia_scores = MIA_MANAGER.run_membership_inference_attack(dataset_name, model_path, use_federated_model)

            for model_name, mia_metric_scores in mia_scores.items():
                print(f" Model name: {model_name} ".center(PRINT_WIDTH, '-'))
                for mia_metric_name, mia_metric_score in mia_metric_scores.items():
                    print(f"{mia_metric_name}: {mia_metric_score}, ", end='')
                print()
