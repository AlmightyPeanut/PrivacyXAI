import os
from collections import OrderedDict

import numpy as np
import torch
import tqdm
from opacus import PrivacyEngine
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader

from .LRClassifier import LRClassifier
from .ModelConfig import ModelConfig
from .NNClassifier import NNClassifier

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
else:
    TORCH_DEVICE = torch.device("cpu")


# These start and end symbols for the parameters of each model have to be numbers
# to work with flower's federated learning
PARAMETER_ARRAY_END_SYMBOL = 2**32 - 1
PARAMETER_ARRAY_LR_SYMBOL = 2**32 - 2
PARAMETER_ARRAY_SVC_SYMBOL = 2**32 - 3
PARAMETER_ARRAY_NN_SYMBOL = 2** 32 - 4


class NoValidModelSpecified(Exception):
    pass


class ModelManager:
    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config

        if not self.config.target_models:
            raise NoValidModelSpecified

        self.privacy_engine = PrivacyEngine()

        if 'LR' in self.config.target_models:
            self.lr_model = LRClassifier(self.config.number_of_features, self.config.number_of_classes).to(TORCH_DEVICE)
            self.lr_loss_function = nn.CrossEntropyLoss()
            self.lr_optimizer = SGD(self.lr_model.parameters(), lr=.1, momentum=0.9)
            self.lr_learning_rate_scheduler = ExponentialLR(self.lr_optimizer, gamma=0.1)

        if 'SVC' in self.config.target_models:
            # TODO: implement svc in torch for more performance
            self.standard_scaler = StandardScaler()
            self.kernel = Nystroem(n_components=self.config.number_of_features)
            self.svc_model = SGDClassifier(loss='log_loss', shuffle=False, warm_start=True)

        if 'NN' in self.config.target_models:
            self.nn_model = NNClassifier(number_of_features=self.config.number_of_features,
                                         number_of_classes=self.config.number_of_classes).to(TORCH_DEVICE)
            self.nn_loss_function = nn.CrossEntropyLoss()
            self.nn_optimizer = SGD(self.nn_model.parameters(), lr=.1, momentum=0.9)
            self.nn_learning_rate_scheduler = ExponentialLR(self.nn_optimizer, gamma=0.1)

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
        for batch_index, batch_data in enumerate(train_data):
            pbar_postfix = []

            if 'LR' in self.config.target_models:
                lr_loss = self._train_lr_model_one_batch(batch_data)
                pbar_postfix.append(f" LR Loss: {lr_loss:.3f}")

            if 'SVC' in self.config.target_models:
                self._train_svc_model_one_epoch(batch_index, batch_data)
                pbar_postfix.append(f" SVC Loss: n/a")

            if 'NN' in self.config.target_models:
                nn_loss = self._train_nn_model_one_batch(batch_data)
                pbar_postfix.append(f" NN Loss: {nn_loss:.3f}")

            pbar.set_postfix_str(','.join(pbar_postfix))
            pbar.update()

        if 'NN' in self.config.target_models:
            self.nn_learning_rate_scheduler.step()

    def _train_lr_model_one_batch(self, batch_data: dict[str, np.ndarray]) -> float:
        self.lr_optimizer.zero_grad()

        outputs = self.lr_model(torch.tensor(batch_data['features'], dtype=torch.float32, device=TORCH_DEVICE))
        loss = self.lr_loss_function(outputs,
                                     torch.tensor(batch_data['classes'], dtype=torch.long, device=TORCH_DEVICE))

        loss.backward()
        self.lr_optimizer.step()

        return loss.item()

    def _train_svc_model_one_epoch(self, batch_index: int, batch_data: dict[str, np.array]) -> float:
        # TODO: research resetting sample counter when training multiple epochs with partial_fit
        scaled_data = self.standard_scaler.fit_transform(batch_data['features'])
        kernel_features = self.kernel.fit_transform(scaled_data)
        if batch_index == 0:
            self.svc_model.partial_fit(kernel_features, batch_data['classes'],
                                       classes=list(range(self.config.number_of_classes)))
        else:
            self.svc_model.partial_fit(kernel_features, batch_data['classes'])

        return .0

    def _train_nn_model_one_batch(self, batch_data: dict[str, np.array]) -> float:
        self.nn_optimizer.zero_grad()

        outputs = self.nn_model(torch.tensor(batch_data['features'], dtype=torch.float32, device=TORCH_DEVICE))
        loss = self.nn_loss_function(outputs,
                                     torch.tensor(batch_data['classes'], dtype=torch.long, device=TORCH_DEVICE))

        loss.backward()
        self.nn_optimizer.step()

        return loss.item()

    def evaluate_target_models(self, test_data: DataLoader) -> dict[str, dict[str, float]]:
        test_data_samples = next(iter(test_data))
        evaluation_results = dict()

        if 'LR' in self.config.target_models:
            with torch.no_grad():
                class_probabilities = self.lr_model(
                    torch.tensor(test_data_samples['features'], dtype=torch.float32, device=TORCH_DEVICE))
            class_probabilities = F.softmax(class_probabilities, dim=-1).cpu().numpy()
            predictions = np.argmax(class_probabilities, axis=1)

            evaluation_results['LR'] = {
                "F1 Score": f1_score(test_data_samples['classes'], predictions, average='micro'),
                "AUC": roc_auc_score(test_data_samples['classes'], class_probabilities, multi_class='ovr',
                                     average='macro')
            }

        if 'SVC' in self.config.target_models:
            scaled_data = self.standard_scaler.fit_transform(test_data_samples['features'])
            kernel_features = self.kernel.fit_transform(scaled_data)
            predictions = self.svc_model.predict(kernel_features)
            evaluation_results['SVC'] = {
                "F1 Score": f1_score(test_data_samples['classes'], predictions, average='micro'),
                "AUC": "n/a"
            }

        if 'NN' in self.config.target_models:
            self.nn_model.eval()
            with torch.no_grad():
                class_probabilities = self.nn_model(
                    torch.tensor(test_data_samples['features'], dtype=torch.float32, device=TORCH_DEVICE))
            class_probabilities = F.softmax(class_probabilities, dim=-1).cpu().numpy()
            predictions = np.argmax(class_probabilities, axis=1)

            evaluation_results['NN'] = {
                "F1 Score": f1_score(test_data_samples['classes'], predictions, average='micro'),
                "AUC": roc_auc_score(test_data_samples['classes'], class_probabilities, multi_class='ovr',
                                     average='macro')
            }

        return evaluation_results

    def get_parameters_of_models(self) -> list[np.array]:
        parameters_of_models = []

        if 'LR' in self.config.target_models:
            parameters_of_models.append(np.array(PARAMETER_ARRAY_LR_SYMBOL))
            parameters_of_models.extend([value.cpu().numpy() for value in self.lr_model.state_dict().values()])
            parameters_of_models.append(np.array(PARAMETER_ARRAY_END_SYMBOL))

        if 'SVC' in self.config.target_models:
            # parameters_of_models['SVC'] = [value.cpu().numpy() for value in self.svc_model.state_dict().values()]
            raise NotImplementedError("Getting parameters from SVC is not supported yet")

        if 'NN' in self.config.target_models:
            parameters_of_models.append(np.array(PARAMETER_ARRAY_NN_SYMBOL))
            parameters_of_models.extend([value.cpu().numpy() for value in self.nn_model.state_dict().values()])
            parameters_of_models.append(np.array(PARAMETER_ARRAY_END_SYMBOL))

        return parameters_of_models

    @staticmethod
    def _pop_model_parameters_from_array(input_array: list[np.array]) -> list[np.array]:
        model_parameters = []

        while np.all(input_array[0] != PARAMETER_ARRAY_END_SYMBOL):
            model_parameters.append(input_array.pop(0))
        input_array.pop(0)

        return model_parameters

    def set_parameters_of_models(self, parameters_of_models: list[np.array]) -> None:
        while parameters_of_models:
            model_symbol = parameters_of_models.pop(0).item()
            model_parameters = self._pop_model_parameters_from_array(parameters_of_models)

            if model_symbol == PARAMETER_ARRAY_LR_SYMBOL:
                parameters_dict = zip(self.lr_model.state_dict().keys(), model_parameters)
                new_state_dict = OrderedDict({key: torch.tensor(value) for key, value in parameters_dict})
                self.lr_model.load_state_dict(new_state_dict)

            if model_symbol == PARAMETER_ARRAY_SVC_SYMBOL:
                raise NotImplementedError("Setting parameters for SVC is not supported")

            if model_symbol == PARAMETER_ARRAY_NN_SYMBOL:
                parameters_dict = zip(self.nn_model.state_dict().keys(), model_parameters)
                new_state_dict = OrderedDict({key: torch.tensor(value) for key, value in parameters_dict})
                self.nn_model.load_state_dict(new_state_dict)

    def privatise_models_and_data(self, data_loader: DataLoader) -> DataLoader:
        new_data_loader = None

        if 'LR' in self.config.target_models:
            self.lr_model, self.lr_optimizer, new_data_loader = self.privacy_engine.make_private(
                module=self.lr_model,
                optimizer=self.lr_optimizer,
                data_loader=data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm
            )

        if 'SVC' in self.config.target_models:
            raise NotImplementedError("Privatising SVC is not supported")

        if 'NN' in self.config.target_models:
            self.nn_model, self.nn_optimizer, new_data_loader = self.privacy_engine.make_private(
                module=self.nn_model,
                optimizer=self.nn_optimizer,
                data_loader=data_loader,
                noise_multiplier=self.config.noise_multiplier,
                max_grad_norm=self.config.max_grad_norm
            )

        if new_data_loader is not None:
            return new_data_loader
        return data_loader

    def save_models(self, path: os.PathLike) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' does not exist")

        if 'LR' in self.config.target_models:
            torch.save(self.lr_model.state_dict(), os.path.join(path, 'lr_model.pth'))

        if 'SVC' in self.config.target_models:
            raise NotImplementedError("Saving SVC model is not supported")

        if 'NN' in self.config.target_models:
            torch.save(self.nn_model.state_dict(), os.path.join(path, 'nn_model.pth'))

    def load_models(self, path: os.PathLike) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' does not exist")

        if 'LR' in self.config.target_models:
            self.lr_model.load_state_dict(torch.load(os.path.join(path, 'lr_model.pth')))

        if 'SVC' in self.config.target_models:
            raise NotImplementedError("Saving SVC model is not supported")

        if 'NN' in self.config.target_models:
            self.nn_model.load_state_dict(torch.load(os.path.join(path, 'nn_model.pth')))

