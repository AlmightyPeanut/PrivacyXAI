import json
import marshal
import pickle
import types
from functools import partial
from logging import INFO
from typing import Callable

import flwr as fl
import numpy as np
import tqdm
from flwr.client import Client
from flwr.common import log
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader

from .FederatedLearningClient import scalar, FederatedLearningClient
from .FederatedLearningConfig import FederatedLearningConfig
from .FederatedLearningStrategy import FederatedLearningStrategy
from experiments.model.ModelManager import ModelManager
from experiments.dataset.DatasetManager import DATASET_MANAGER
from ..utils import MODEL_CHECKPOINTS_PATH, RESULTS_PATH, PRINT_WIDTH

fl.common.logger.configure(identifier='FederatedLearningManager', filename='federated_learning.log')


def run_task(*args, **kwargs):
    marshaled = kwargs.pop('marshaled_func')
    func = marshal.loads(marshaled)
    pickled_closure = kwargs.pop('pickled_closure')
    pickled_closure = pickle.loads(pickled_closure)

    restored_function = types.FunctionType(func, globals(), closure=create_closure(func, pickled_closure))
    return restored_function(*args, **kwargs)


def create_closure(func, original_closure):
    indent = " " * 4
    closure_vars_def = f"\n{indent}".join(f"{name}=None" for name in func.co_freevars)
    closure_vars_ref = ",".join(func.co_freevars)
    dynamic_closure = "create_dynamic_closure"
    s = (f"""
def {dynamic_closure}():
    {closure_vars_def}
    def internal():
        {closure_vars_ref}
    return internal.__closure__
    """)
    exec(s)
    created_closure = locals()[dynamic_closure]()
    for closure_var, value in zip(created_closure, original_closure):
        closure_var.cell_contents = value
    return created_closure


def wrap_task(pool: Pool, task, generator):
    closure = task.__closure__
    pickled_closure = pickle.dumps(tuple(x.cell_contents for x in closure))
    marshaled_func = marshal.dumps(task.__code__)
    with tqdm.tqdm(total=len(generator)) as pbar:
        for _ in pool.imap_unordered(partial(run_task, marshaled_func=marshaled_func, pickled_closure=pickled_closure),
                                     generator):
            pbar.update()


def get_evaluate_fn(test_data_loader: DataLoader, number_of_features: int, number_of_classes: int) -> Callable:
    """ Server evaluation function. Does not return a loss value."""

    def evaluate(server_round: int, parameters_of_models: list[np.array],
                 config: dict[str, scalar]) -> (float, dict[dict[str, scalar]]):
        model_manager = ModelManager(number_of_features, number_of_classes)
        model_manager.set_parameters_of_models(parameters_of_models)
        evaluation_scores = model_manager.evaluate_target_models(test_data_loader, 0, {}, False)

        return .0, evaluation_scores

    return evaluate


def generate_client_fn(train_data_loaders: list[DataLoader],
                       number_of_features: int,
                       number_of_classes: int,
                       use_differential_privacy: bool,
                       epsilon: float,
                       client_training_epochs: int) -> Callable:
    def generate_client(client_id: str) -> Client:
        client_id = int(client_id)
        return FederatedLearningClient(
            train_data_loaders[client_id],
            number_of_features,
            number_of_classes,
            use_differential_privacy,
            epsilon,
            client_training_epochs,
        ).to_client()

    return generate_client


def _run_federated_learning(config: FederatedLearningConfig, number_of_clients: int,
                            client_fn_callback: Callable,
                            test_data: DataLoader,
                            number_of_features: int, number_of_classes: int,
                            use_differential_privacy: bool,
                            epsilon: float,
                            fold_index: int,
                            dataset_name: str, ):
    federated_learning_strategy = FederatedLearningStrategy(
        fraction_fit=config.fraction_fit,
        fraction_evaluate=config.fraction_evaluate,
        min_available_clients=number_of_clients,
        evaluate_fn=get_evaluate_fn(test_data, number_of_features, number_of_classes),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn_callback,
        num_clients=number_of_clients,
        config=fl.server.ServerConfig(num_rounds=config.num_rounds),
        strategy=federated_learning_strategy,
        client_resources=config.client_resources,
        # ray_init_args={"local_mode": True},
    )

    fold_result = history.metrics_centralized
    file_name = f'fl_model_metrics_fl=True_clients={number_of_clients}_rounds={config.num_rounds}_fold={fold_index}'
    if use_differential_privacy:
        file_name += f'_privatised_epsilon={epsilon}'
    file_name += '.json'
    with open(RESULTS_PATH / file_name, 'w') as f:
        json.dump(fold_result, f)

    results_message = f" FL training results for {dataset_name} ".center(PRINT_WIDTH, '#')
    results_message += f"Client training results".center(PRINT_WIDTH, '_')
    for model_name, iteration_results in fold_result.items():
        results_message += f" Model name: {model_name} ".center(PRINT_WIDTH, '-')
        for iteration, metric_scores in iteration_results:
            results_message += f"Iteration {iteration}: "
            for metric_name, metric_score in metric_scores.items():
                results_message += f"{metric_name}: {metric_score}, "
            results_message += '\n'
        results_message += '\n'
    results_message += '\n'
    log(INFO, results_message)

    # save server model
    model_manager = ModelManager(number_of_features, number_of_classes)
    model_manager.set_parameters_of_models(federated_learning_strategy.get_parameters_of_models())
    fl_parameters = {
        "fl": True,
        "fl_clients": number_of_clients,
        "fl_rounds": config.num_rounds,
        "privatised": use_differential_privacy,
        "epsilon": epsilon,
        "fold": fold_index,
    }
    model_manager.save_models(MODEL_CHECKPOINTS_PATH / 'fl_server_model', fl_parameters)


class FederatedLearningManager:
    def __init__(self,
                 number_of_clients: int,
                 epsilons: list[float],
                 config: FederatedLearningConfig = FederatedLearningConfig()):
        self.config = config
        self.number_of_clients = number_of_clients
        self.epsilons = epsilons

    def start_simulation(self, dataset_name: str) -> None:
        number_of_features = DATASET_MANAGER.get_number_of_features(dataset_name)
        number_of_classes = DATASET_MANAGER.get_number_of_classes(dataset_name)

        for fold_index, (train_data, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
            def training_task(dp_settings: tuple[bool, float]):
                train_data_loaders = DATASET_MANAGER.split_data_for_federated_learning(train_data,
                                                                                       self.number_of_clients)
                client_fn_callback = generate_client_fn(train_data_loaders, number_of_features, number_of_classes,
                                                        dp_settings[0], dp_settings[1],
                                                        self.config.client_training_epochs)
                _run_federated_learning(
                    self.config, self.number_of_clients,
                    client_fn_callback,
                    test_data,
                    number_of_features, number_of_classes,
                    dp_settings[0],
                    dp_settings[1],
                    fold_index,
                    dataset_name, )

            _dp_settings = [(True, epsilon) for epsilon in self.epsilons]
            _dp_settings.append((False, .0))
            with Pool(len(self.epsilons) + 1) as pool:
                wrap_task(pool, training_task, _dp_settings)
