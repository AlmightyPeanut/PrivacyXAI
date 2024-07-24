import json
import os
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from art.estimators.classification import PyTorchClassifier
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader

from .LRClassifier import LRClassifier
from .ModelConfig import ModelConfig
from .NNClassifier import NNClassifier
from ..utils import RESULTS_PATH, PRINT_WIDTH

torch.manual_seed(42)

# These start and end symbols for the parameters of each model have to be numbers
# to work with flower's federated learning
PARAMETER_ARRAY_END_SYMBOL = 2 ** 32 - 1
PARAMETER_ARRAY_LR_SYMBOL = 2 ** 32 - 5
PARAMETER_ARRAY_NN_SYMBOL = 2 ** 32 - 10


class NoValidModelSpecified(Exception):
    pass


class ModelManager:
    def __init__(self, number_of_features, number_of_classes, config: ModelConfig = ModelConfig(),
                 use_nn_regularisation=False):
        if torch.cuda.is_available():
            self.TORCH_DEVICE = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.TORCH_DEVICE = torch.device("mps")
        else:
            self.TORCH_DEVICE = torch.device("cpu")

        self.number_of_features = number_of_features
        self.number_of_classes = number_of_classes
        self.config = config
        self.use_private_models = False

        if not self.config.target_models:
            raise NoValidModelSpecified

        if 'LR' in self.config.target_models:
            self.lr_model = LRClassifier(self.number_of_features, self.number_of_classes).to(self.TORCH_DEVICE)
            self.lr_loss_function = nn.BCELoss()
            if self.number_of_classes > 1:
                self.lr_loss_function = nn.CrossEntropyLoss()
            self.lr_optimizer = SGD(self.lr_model.parameters(), lr=0.001)
            self.lr_privacy_engine = PrivacyEngine(accountant='rdp')

        if 'NN' in self.config.target_models:
            self.nn_model = NNClassifier(number_of_features=self.number_of_features,
                                         number_of_classes=self.number_of_classes).to(self.TORCH_DEVICE)
            self.nn_loss_function = nn.BCELoss()
            if self.number_of_classes > 1:
                self.nn_loss_function = nn.CrossEntropyLoss()
            if use_nn_regularisation:
                self.nn_optimizer = AdamW(self.nn_model.parameters(), lr=0.001, weight_decay=1.)
            else:
                self.nn_optimizer = AdamW(self.nn_model.parameters(), lr=0.001)
            self.nn_privacy_engine = PrivacyEngine(accountant='rdp')

    def train_target_models(self, train_data: DataLoader) -> None:
        for epoch_index in range(self.config.number_of_epochs):
            # TODO: print tqdm to correct log output for flower -> maybe only show progress centralised?
            with tqdm.tqdm(total=len(train_data), desc=f'Epoch {epoch_index + 1}/{self.config.number_of_epochs}',
                           unit=' batches') as pbar:
                self._train_target_models_one_epoch(train_data, pbar)

    def _train_target_models_one_epoch(self, train_data: DataLoader, pbar: tqdm.tqdm):
        if 'LR' in self.config.target_models:
            self.lr_model.train()

        if 'NN' in self.config.target_models:
            self.nn_model.train()

        batch_data: dict[str, np.array]
        lr_epoch_loss = .0
        nn_epoch_loss = .0
        for batch_index, batch_data in enumerate(train_data):
            train_data_x = torch.tensor(batch_data['features'], dtype=torch.float32).to(self.TORCH_DEVICE)
            train_data_y = torch.tensor(batch_data['classes'], dtype=torch.float32).to(self.TORCH_DEVICE)

            pbar_postfix = []
            epsilon_postfix = ''
            if 'LR' in self.config.target_models:
                lr_loss = self._train_lr_model_one_batch(train_data_x, train_data_y)
                lr_epoch_loss += lr_loss
                pbar_postfix.append(f" LR Loss: {lr_loss:.3f} (Epoch avg.: {lr_epoch_loss / (batch_index + 1):.3f})")
                if self.use_private_models:
                    epsilon_postfix = f" ε = {self.lr_privacy_engine.get_epsilon(self.config.dp_target_delta)}"

            if 'NN' in self.config.target_models:
                nn_loss = self._train_nn_model_one_batch(train_data_x, train_data_y)
                nn_epoch_loss += nn_loss
                pbar_postfix.append(f" NN Loss: {nn_loss:.3f} (Epoch avg.: {nn_epoch_loss / (batch_index + 1):.3f})")
                if self.use_private_models:
                    epsilon_postfix = f" ε = {self.nn_privacy_engine.get_epsilon(self.config.dp_target_delta)}"

            if epsilon_postfix != '':
                pbar_postfix.append(epsilon_postfix)

            pbar.set_postfix_str(','.join(pbar_postfix))
            pbar.update()

    def _train_lr_model_one_batch(self, train_data_x: torch.Tensor, train_data_y: torch. Tensor) -> float:
        outputs = self.lr_model(train_data_x)
        loss = self.lr_loss_function(outputs, train_data_y)

        self.lr_optimizer.zero_grad()
        loss.backward()
        self.lr_optimizer.step()

        return loss.item()

    def _train_nn_model_one_batch(self, train_data_x: torch.Tensor, train_data_y: torch. Tensor) -> float:
        outputs = self.nn_model(train_data_x)
        loss = self.nn_loss_function(outputs, train_data_y)

        self.nn_optimizer.zero_grad()
        loss.backward()
        self.nn_optimizer.step()

        return loss.item()

    def evaluate_target_models(self, test_data: DataLoader, fold_index: int, model_parameters: dict,
                               save_results: bool = True) -> dict[str, dict[str, float]]:
        test_data_samples = next(iter(test_data))
        evaluation_results = dict()

        prediction_logits = self.predict_one_batch_logits(test_data_samples['features'])

        if 'LR' in prediction_logits:
            evaluation_results['LR'] = self._calculate_training_metrics(prediction_logits['LR'],
                                                                        test_data_samples['classes'])

        if 'NN' in self.config.target_models:
            evaluation_results['NN'] = self._calculate_training_metrics(prediction_logits['NN'],
                                                                        test_data_samples['classes'])

        if save_results:
            file_name = f"centralised_model_metrics_fold={fold_index}"
            if "privatised" in model_parameters and model_parameters["privatised"]:
                file_name += f"_privatised_eps={model_parameters['epsilon']}_delta={self.config.dp_target_delta}_max_grad_norm={self.config.dp_max_grad_norm}"
            if "fl" in model_parameters and model_parameters["fl"]:
                file_name += f"_fl_clients={model_parameters['fl_clients']}_rounds={model_parameters['fl_rounds']}"
            file_name = f'{file_name}.json'
            with open(RESULTS_PATH / file_name, 'w') as f:
                json.dump(evaluation_results, f)

            print(f" Training results for fold {fold_index} ".center(PRINT_WIDTH, '#'))
            for model_name, metric_scores in evaluation_results.items():
                print(f" Model name: {model_name} ".center(PRINT_WIDTH, '_'))
                for metric_name, metric_score in metric_scores.items():
                    print(f"{metric_name}: {metric_score}")

        return evaluation_results

    def _calculate_training_metrics(self, prediction_logits: np.array, targets: np.array) -> dict[str, float]:
        if self.number_of_classes == 1:
            prediction_classes = np.round(prediction_logits, decimals=0)
        else:
            prediction_classes = np.argmax(prediction_logits, axis=1)
        evaluation_results = {
            "AUROC": roc_auc_score(targets, prediction_logits,
                                   multi_class='ovr', average='macro'),
            "Acc": accuracy_score(targets, prediction_classes),
            "Macro F1 Score": f1_score(targets, prediction_classes, average='macro'),
            "Binary F1 Score": f1_score(targets, prediction_classes, average='binary'),
        }
        return evaluation_results

    def get_parameters_of_models(self) -> list[np.array]:
        parameters_of_models = []

        if 'LR' in self.config.target_models:
            parameters_of_models.append(np.array(PARAMETER_ARRAY_LR_SYMBOL))
            parameters_of_models.extend([value.cpu().numpy() for value in self.lr_model.state_dict().values()])
            parameters_of_models.append(np.array(PARAMETER_ARRAY_END_SYMBOL))

        if 'NN' in self.config.target_models:
            parameters_of_models.append(np.array(PARAMETER_ARRAY_NN_SYMBOL))
            parameters_of_models.extend([value.cpu().numpy() for value in self.nn_model.state_dict().values()])
            parameters_of_models.append(np.array(PARAMETER_ARRAY_END_SYMBOL))

        return parameters_of_models

    @staticmethod
    def _pop_model_parameters_from_array(input_array: list[np.array]) -> list[np.array]:
        model_parameters = []

        while len(input_array[0].shape) != 0:
            model_parameters.append(input_array.pop(0))
        input_array.pop(0)

        return model_parameters

    def set_parameters_of_models(self, parameters_of_models: list[np.array]) -> None:
        while parameters_of_models:
            model_symbol = int(parameters_of_models.pop(0).item())
            model_parameters = self._pop_model_parameters_from_array(parameters_of_models)

            if abs(model_symbol - PARAMETER_ARRAY_LR_SYMBOL) < 2:
                parameters_dict = zip(self.lr_model.state_dict().keys(), model_parameters)
                new_state_dict = OrderedDict({key: torch.tensor(value) for key, value in parameters_dict})
                self.lr_model.load_state_dict(new_state_dict)
                continue

            if abs(model_symbol - PARAMETER_ARRAY_NN_SYMBOL) < 2:
                parameters_dict = zip(self.nn_model.state_dict().keys(), model_parameters)
                new_state_dict = OrderedDict({key: torch.tensor(value) for key, value in parameters_dict})
                self.nn_model.load_state_dict(new_state_dict)
                continue

            raise ValueError(f"Unknown model symbol {model_symbol}")

    def privatise_models_and_data(self, data_loader: DataLoader, epsilon: float):
        print(f"Privatising with eps={epsilon}")
        self.use_private_models = True

        # new_data_loader = None
        if 'LR' in self.config.target_models:
            self.lr_model, self.lr_optimizer, data_loader = self.lr_privacy_engine.make_private_with_epsilon(
                module=self.lr_model,
                optimizer=self.lr_optimizer,
                data_loader=data_loader,
                target_epsilon=epsilon,
                target_delta=self.config.dp_target_delta,
                epochs=self.config.number_of_epochs,
                max_grad_norm=self.config.dp_max_grad_norm,
            )

        if 'NN' in self.config.target_models:
            self.nn_model, self.nn_optimizer, data_loader = self.nn_privacy_engine.make_private_with_epsilon(
                module=self.nn_model,
                optimizer=self.nn_optimizer,
                data_loader=data_loader,
                target_epsilon=epsilon,
                target_delta=self.config.dp_target_delta,
                epochs=self.config.number_of_epochs,
                max_grad_norm=self.config.dp_max_grad_norm,
            )

        return data_loader

    def save_models(self, model_folder_path: os.PathLike, parameters: dict) -> None:
        if not os.path.exists(model_folder_path):
            raise FileNotFoundError(f"'{model_folder_path}' does not exist")

        model_parameters = f"fold={parameters['fold']}_epochs={self.config.number_of_epochs}"
        if "privatised" in parameters and parameters["privatised"]:
            model_parameters += f"_privatised_eps={parameters['epsilon']}_delta={self.config.dp_target_delta}_max_grad_norm={self.config.dp_max_grad_norm}"
        if "fl" in parameters and parameters["fl"]:
            model_parameters += f"_fl_clients={parameters['fl_clients']}_rounds={parameters['fl_rounds']}"

        if 'LR' in self.config.target_models:
            torch.save(self.lr_model.state_dict(), os.path.join(model_folder_path, f'lr_model_{model_parameters}.pth'))

        if 'NN' in self.config.target_models:
            torch.save(self.nn_model.state_dict(), os.path.join(model_folder_path, f'nn_model_{model_parameters}.pth'))

    def load_model(self, model_file_path: os.PathLike) -> None:
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"'{model_file_path}' does not exist")

        if str(model_file_path).startswith('lr'):
            self.lr_model.load_state_dict(torch.load(os.path.join(model_file_path)))

        if str(model_file_path).startswith('nn'):
            self.nn_model.load_state_dict(torch.load(os.path.join(model_file_path)))

    def prepare_models_for_mia(self) -> dict[str, PyTorchClassifier]:
        models = {}

        if 'LR' in self.config.target_models:
            models['LR'] = PyTorchClassifier(
                model=self.lr_model,
                loss=self.lr_loss_function,
                input_shape=(1, self.number_of_features),
                nb_classes=self.number_of_classes,
            )

        if 'NN' in self.config.target_models:
            models['NN'] = PyTorchClassifier(
                model=self.nn_model,
                loss=self.nn_loss_function,
                input_shape=(1, self.number_of_features),
                nb_classes=self.number_of_classes,
            )

        return models

    def predict_one_batch_logits(self, data: np.array) -> dict[str, np.ndarray]:
        test_data = torch.tensor(data, dtype=torch.float32, device=self.TORCH_DEVICE)

        prediction_logits = {}
        if 'LR' in self.config.target_models:
            self.lr_model.eval()
            with torch.no_grad():
                class_probabilities = self.lr_model(test_data).detach().cpu().numpy()
            prediction_logits['LR'] = class_probabilities

        if 'NN' in self.config.target_models:
            self.nn_model.eval()
            with torch.no_grad():
                class_probabilities = self.nn_model(test_data).detach().cpu().numpy()
            prediction_logits['NN'] = class_probabilities

        return prediction_logits

    def predict_one_batch_classes(self, data: np.array) -> dict[str, np.ndarray]:
        predictions = self.predict_one_batch_logits(data)

        if 'LR' in predictions:
            predictions['LR'] = np.argmax(predictions['LR'], axis=-1)

        if 'NN' in predictions:
            predictions['NN'] = np.argmax(predictions['NN'], axis=-1)

        return predictions

    def __getitem__(self, item: str):
        if item == 'LR':
            return self.lr_model

        if item == 'NN':
            return self.nn_model

        raise KeyError(item)
