import marshal
import os
import pickle
import types
from functools import partial

import torch
import tqdm
from torch.multiprocessing import Pool
from torch.utils.data import DataLoader

from experiments.dataset.DatasetManager import DATASET_MANAGER
from experiments.model.ModelManager import ModelManager
from experiments.federated_learning.FederatedLearningManager import FederatedLearningManager
from experiments.utils import MODEL_CHECKPOINTS_PATH, PRINT_WIDTH
from experiments.xai.XAIManager import XAIManager
from experiments.mia.MIAManager import run_membership_inference_attack


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


def wrap_single_task(task, *args):
    closure = task.__closure__
    pickled_closure = pickle.dumps(tuple(x.cell_contents for x in closure))
    marshaled_func = marshal.dumps(task.__code__)
    return torch.multiprocessing.spawn(
        partial(run_task, marshaled_func=marshaled_func, pickled_closure=pickled_closure),
        args=args,
        join=False
    )


def _run_centralised_training(dataset_name: str, train_data: DataLoader, test_data: DataLoader, fold_index: int,
                              use_differential_privacy: bool, epsilon: float = .0):
    model_manager = ModelManager(DATASET_MANAGER.get_number_of_features(dataset_name),
                                 DATASET_MANAGER.get_number_of_classes(dataset_name), use_nn_regularisation=True)
    if use_differential_privacy:
        train_data = model_manager.privatise_models_and_data(train_data, epsilon=epsilon)

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


def _run_centralised_model(dataset_name: str, epsilons: list[float]):
    for fold_index, (train_data, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
        def training_task(dp_settings: tuple[bool, float]):
            _run_centralised_training(dataset_name, train_data, test_data, fold_index,
                                      dp_settings[0], dp_settings[1])

        _dp_settings = [(True, epsilon) for epsilon in epsilons]
        _dp_settings.append((False, .0))
        with Pool(len(epsilons) + 1) as pool:
            wrap_task(pool, training_task, _dp_settings)


class Experiment:
    def __init__(self, number_of_clients: list[int], epsilons: list[float]):
        self.number_of_clients = number_of_clients
        self.epsilons = epsilons

    def run_model_training(self, federated_learning: bool = True, centralised_model: bool = True):
        for dataset_name in DATASET_MANAGER.datasets:
            processes = []

            if centralised_model:
                def _centralised_training_task(task_id, epsilons: list[float]):
                    _run_centralised_model(dataset_name, epsilons)

                processes.append(wrap_single_task(_centralised_training_task, self.epsilons))

            if federated_learning:
                for number_of_clients in self.number_of_clients:
                    def _federated_training_task(task_id, epsilons: list[float]):
                        fl_manager = FederatedLearningManager(number_of_clients, epsilons)
                        fl_manager.start_simulation(dataset_name)

                    _federated_training_task(0, self.epsilons)

                    processes.append(wrap_single_task(_federated_training_task, self.epsilons))

            for process in processes:
                process.join()

    @staticmethod
    def _get_model_paths(use_centralised_model: bool, use_federated_model: bool) -> list[os.PathLike]:
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
        return model_paths

    def run_xai_evaluation(self, use_federated_model=True, use_centralised_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" XAI evaluation on {dataset_name} ".center(PRINT_WIDTH, '#'))

            model_paths = self._get_model_paths(use_centralised_model, use_federated_model)

            for fold_index, (_, test_data) in DATASET_MANAGER.get_data_folds(dataset_name):
                fold_models = [p for p in model_paths if f"fold={fold_index}" in str(p)]
                XAIManager().evaluate_explanations(test_data,
                                                   DATASET_MANAGER.get_number_of_features(dataset_name),
                                                   DATASET_MANAGER.get_number_of_classes(dataset_name),
                                                   fold_models)

    def run_mia(self, use_federated_model=True, use_centralised_model=True):
        for dataset_name in DATASET_MANAGER.datasets:
            print(f" MIA on {dataset_name} ".center(PRINT_WIDTH, '#'))

            model_paths = self._get_model_paths(use_centralised_model, use_federated_model)
            run_membership_inference_attack(dataset_name,
                                            DATASET_MANAGER.get_number_of_features(dataset_name),
                                            DATASET_MANAGER.get_number_of_classes(dataset_name),
                                            model_paths)
