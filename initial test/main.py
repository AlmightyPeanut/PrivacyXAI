from collections import OrderedDict
from typing import Callable, List, Tuple, Union, Optional, Dict

import flwr as fl
import numpy as np
import quantus
import torch
import torch.nn as nn
import torch.nn.functional as F
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier

from art.utils import load_iris
from flwr.common import Scalar, NDArrays, FitRes, Parameters
from flwr.server.client_proxy import ClientProxy
from opacus import PrivacyEngine
from torch.utils.data import DataLoader, TensorDataset, random_split


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=output_size)

    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


def prepare_dataset(n_partitions: int, batch_size: int = 32, validation_ratio: float = 0.1) -> [DataLoader, DataLoader,
                                                                                                DataLoader, int, int]:
    (x_train, y_train), (x_test, y_test), _, _ = load_iris()
    training_dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    training_set_length = len(x_train) // n_partitions
    partition_length = [training_set_length] * n_partitions
    for partition_index in range(len(x_train) - n_partitions * training_set_length):
        partition_length[partition_index % n_partitions] += 1

    training_sets = random_split(training_dataset, partition_length, torch.Generator().manual_seed(42))
    training_data_loaders = []
    validation_data_loaders = []
    for training_set in training_sets:
        n_total_samples = len(training_set)
        n_validation_samples = max(int(validation_ratio * n_total_samples), 1)
        n_training_samples = n_total_samples - n_validation_samples

        local_training_set, local_validation_dataset = random_split(
            training_set,
            [n_training_samples, n_validation_samples],
            torch.Generator().manual_seed(42))

        training_data_loaders.append(DataLoader(local_training_set, batch_size=batch_size, shuffle=True))
        validation_data_loaders.append(DataLoader(local_validation_dataset, batch_size=batch_size, shuffle=True))

    test_data_loader = DataLoader(test_dataset, batch_size=len(training_dataset), shuffle=True)

    return training_data_loaders, validation_data_loaders, test_data_loader, x_train.shape[1], y_train.shape[1]


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, train_data_loader: DataLoader, validation_data_loader: DataLoader,
                 model_input_size: int, model_output_size: int) -> None:
        super().__init__()

        model = Net(model_input_size, model_output_size)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        # privatise
        privacy_engine = PrivacyEngine()
        self.model, self.optimizer, self.train_data_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_data_loader,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
        )

        self.validation_data_loader = validation_data_loader

    def set_parameters(self, parameters) -> None:
        parameters_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.Tensor(value) for key, value in parameters_dict})
        self.model.load_state_dict(state_dict)

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        return [value.cpu().numpy() for _, value in self.model.state_dict().items()]

    def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(epochs := 5):
            for features, labels in self.train_data_loader:
                self.optimizer.zero_grad()
                loss = criterion(self.model(features), labels)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters({}), len(self.train_data_loader), {}

    def evaluate(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        self.set_parameters(parameters)

        criterion = nn.CrossEntropyLoss()
        correct_predictions = 0
        loss = 0.0
        self.model.eval()
        with torch.no_grad():
            for features, labels in self.validation_data_loader:
                output = self.model(features)
                loss += criterion(output, labels).item()
                _, predictions = torch.max(output.data, dim=1)
                _, true_labels = torch.max(labels.data, dim=1)
                correct_predictions += (predictions == true_labels).sum().item()
        accuracy = correct_predictions / len(self.validation_data_loader.dataset)

        return float(loss), len(self.validation_data_loader), {"accuracy": accuracy}


def get_evaluate_fn(test_data_loader: DataLoader, model_input_size: int, model_output_size: int) -> Callable:
    def evaluate(server_round: int, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[float, dict[str, Scalar]]:
        model = Net(model_input_size, model_output_size)

        parameter_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({key: torch.Tensor(value) for key, value in parameter_dict})
        model.load_state_dict(state_dict)

        criterion = nn.CrossEntropyLoss()
        correct_predictions = 0
        loss = 0.0
        model.eval()
        with torch.no_grad():
            for features, labels in test_data_loader:
                output = model(features)
                loss += criterion(output, labels).item()
                _, predictions = torch.max(output.data, dim=1)
                _, true_labels = torch.max(labels.data, dim=1)
                correct_predictions += (predictions == true_labels).sum().item()
        accuracy = correct_predictions / len(test_data_loader.dataset)

        return float(loss), {"accuracy": accuracy}

    return evaluate


def generate_client_fn(train_data_loaders: list[DataLoader], validation_data_loaders: list[DataLoader],
                       model_input_size: int, model_output_size: int) -> Callable:
    def client_fn(client_id: str) -> FlowerClient:
        return FlowerClient(
            train_data_loader=train_data_loaders[int(client_id)],
            validation_data_loader=validation_data_loaders[int(client_id)],
            model_input_size=model_input_size,
            model_output_size=model_output_size,
        )

    return client_fn


class SaveModelFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, model_input_size: int, model_output_size: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model_state_dict_keys = Net(model_input_size, model_output_size).state_dict().keys()
        self._state_dict = None

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Updating round {server_round} aggregated_parameters...")

            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

            params_dict = zip(self._model_state_dict_keys, aggregated_ndarrays)
            self._state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

        return aggregated_parameters, aggregated_metrics

    def get_model_state_dict(self) -> OrderedDict:
        return self._state_dict


def membership_inference_attack(target_model: PyTorchClassifier,
                                used_in_training_predictions: np.ndarray, used_in_training_y: np.ndarray,
                                not_used_in_training_predictions: np.ndarray, not_used_in_training_y: np.ndarray,
                                records_to_attack_predictions: np.ndarray,
                                records_to_attack_true_labels: np.ndarray) -> None:
    attack = MembershipInferenceBlackBox(estimator=target_model)
    attack.fit(x=None, y=used_in_training_y, pred=used_in_training_predictions,
               test_x=None, test_y=not_used_in_training_y, test_pred=not_used_in_training_predictions)

    membership_status = attack.infer(x=None, y=records_to_attack_true_labels,
                                     pred=records_to_attack_predictions)
    print("Accuracy of MIA: {:.2f}%".format(np.mean(membership_status == records_to_attack_true_labels) * 100))


def main():
    n_clients = 5
    (training_data_loaders, validation_data_loaders, test_data_loader,
     n_features, n_classes) = prepare_dataset(n_partitions=n_clients)
    print("Number of local training examples: {}".format(len(training_data_loaders[0].dataset)))

    federated_learning_strategy = SaveModelFedAvg(
        model_input_size=n_features,
        model_output_size=n_classes,
        fraction_fit=0.5,  # ratio of clients selected to do local training
        fraction_evaluate=0.5,  # ratio of clients selected to assess the global model
        min_available_clients=n_clients,
        evaluate_fn=get_evaluate_fn(test_data_loader, n_features, n_classes)
    )

    client_fn_callback = generate_client_fn(training_data_loaders, validation_data_loaders, n_features, n_classes)

    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=federated_learning_strategy,
    )

    print("Global accuracy: {}".format(history.metrics_centralized["accuracy"][-1][1]))

    server_model = Net(n_features, n_classes)
    server_model.load_state_dict(federated_learning_strategy.get_model_state_dict(), strict=True)
    server_model.eval()

    # TODO: use privatised data here?
    full_train_dataset = torch.utils.data.ConcatDataset([loader.dataset for loader in training_data_loaders])
    full_train_dataloader = DataLoader(full_train_dataset, batch_size=len(full_train_dataset))
    all_training_data_x, all_training_data_y = next(iter(full_train_dataloader))
    all_training_data_x = all_training_data_x.numpy()
    all_training_data_y = all_training_data_y.numpy()

    # Possible XAI methods: 'GradientShap', 'IntegratedGradients', 'DeepLift', 'DeepLiftShap', 'InputXGradient',
    # 'Saliency', 'FeatureAblation', 'Deconvolution', 'FeaturePermutation', 'Lime', 'KernelShap', 'LRP', 'Gradient',
    # 'Occlusion', 'LayerGradCam', 'GuidedGradCam', 'LayerConductance', 'LayerActivation', 'InternalInfluence',
    # 'LayerGradientXActivation', 'Control Var. Sobel Filter', 'Control Var. Constant', 'Control Var. Random Uniform'
    xai_metric = quantus.MaxSensitivity()
    xai_scores = xai_metric(server_model, all_training_data_x, np.argmax(all_training_data_y, axis=1),
                            explain_func=quantus.explain, explain_func_kwargs={"method": "Lime"})

    print("Max sensitivity: {}".format(max(xai_scores)))

    classifier = PyTorchClassifier(
        model=server_model,
        loss=nn.CrossEntropyLoss(),
        input_shape=(1, n_features),
        nb_classes=n_classes,
    )

    train_predictions = classifier.predict(all_training_data_x)
    test_data_x, test_data_y = next(iter(test_data_loader))
    test_predictions = classifier.predict(test_data_x)

    membership_inference_attack(classifier, train_predictions, all_training_data_y, test_predictions, test_data_y,
                                classifier.predict(np.concatenate([all_training_data_x, test_data_x], axis=0)),
                                np.concatenate([np.ones(all_training_data_y.shape[0]),
                                                np.zeros(test_data_y.shape[0])], axis=0))


if __name__ == '__main__':
    main()
