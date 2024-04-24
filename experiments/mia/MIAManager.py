import os

import numpy as np
import tqdm
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from torch.utils.data import DataLoader

from experiments.model.ModelManager import ModelManager
from experiments.dataset.DatasetManager import DATASET_MANAGER
from experiments.utils.Singleton import Singleton


class MIAManager(metaclass=Singleton):
    def __init__(self):
        self.attack_models = {}

    def prepare_predictions_for_attack(self, data: DataLoader, model_manager: ModelManager) -> (
            dict[str, np.ndarray], np.ndarray, np.ndarray
    ):
        data_model_predictions = {}
        data_true_classes = []
        data_train_test_indicator = []

        for batch_data in tqdm.tqdm(data, unit='batch'):
            predictions = model_manager.predict_one_batch_logits(batch_data['features'])
            for model_name, prediction in predictions.items():
                if model_name not in data_model_predictions:
                    data_model_predictions[model_name] = []
                data_model_predictions[model_name].extend(prediction)

            data_true_classes.extend(batch_data['classes'])
            data_train_test_indicator.extend(batch_data['train_test_indicator'])

        for model_name, predictions in data_model_predictions.items():
            data_model_predictions[model_name] = np.array(predictions)
        data_true_classes = np.array(data_true_classes)
        data_train_test_indicator = np.array(data_train_test_indicator)

        return data_model_predictions, data_true_classes, data_train_test_indicator

    def run_membership_inference_attack(
            self, dataset_name: str, model_folder: os.PathLike, use_federated_learning_data: bool
    ) -> dict[str, dict[str, float]]:
        model_manager = ModelManager()
        model_manager.load_models(model_folder)

        print("Preparing train data...")
        target_model_train_predictions, target_model_train_true_classes, _ = (
            self.prepare_predictions_for_attack(
                DATASET_MANAGER.get_attacker_train_data(dataset_name, use_federated_learning_data), model_manager)
        )

        print("Preparing test data...")
        target_model_test_predictions, target_model_test_true_classes, _ = (
            self.prepare_predictions_for_attack(
                DATASET_MANAGER.get_attacker_test_data(dataset_name, use_federated_learning_data), model_manager)
        )

        print("Preparing records to attack...")
        records_to_attack_predictions, records_to_attack_true_classes, records_to_attack_train_test_indicator = (
            self.prepare_predictions_for_attack(
                DATASET_MANAGER.get_records_to_attack(dataset_name, use_federated_learning_data), model_manager)
        )

        target_models = model_manager.prepare_models_for_mia()

        mia_scores = {}
        for target_model_name, target_model in target_models.items():
            print(f"Attacking model {target_model_name}")
            attack = MembershipInferenceBlackBox(estimator=target_model,
                                                 nn_model_epochs=5,
                                                 nn_model_batch_size=4096)

            attack.fit(
                x=None, pred=target_model_train_predictions[target_model_name],
                y=target_model_train_true_classes,
                test_x=None, test_pred=target_model_test_predictions[target_model_name],
                test_y=target_model_test_true_classes,
            )

            membership_status = attack.infer(
                x=None, pred=records_to_attack_predictions[target_model_name],
                y=records_to_attack_true_classes
            )

            mia_scores[target_model_name] = {
                "Accuracy": np.mean(membership_status == records_to_attack_train_test_indicator)
            }

        return mia_scores


MIA_MANAGER = MIAManager()
