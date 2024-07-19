import json
import marshal
import os
import pickle
import re
import types
from functools import partial

import numpy as np
import torch
import tqdm
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox, ShadowModels
from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from torch import multiprocessing
from torch.multiprocessing import Pool

from experiments.mia.MIAConfig import MIAConfig
from experiments.model.LRClassifier import LRClassifier
from experiments.dataset.DatasetManager import DATASET_MANAGER
from experiments.model.NNClassifier import NNClassifier
from experiments.utils import remove_model_prefix, RESULTS_PATH


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


def _run_attack(target_model_path: os.PathLike, fold_index: int,
                number_of_features: int, number_of_classes: int,
                target_model_train_data: dict, target_model_test_data: dict,
                shadow_model_data: dict) -> None:
    if torch.cuda.is_available():
        gpu_id = np.random.randint(0, torch.cuda.device_count())
        torch.cuda.set_device(gpu_id)

    config = MIAConfig()

    if 'lr_model' in str(target_model_path):
        target_model = LRClassifier(number_of_features, number_of_classes)
        shadow_model = LRClassifier(number_of_features, number_of_classes)
        shadow_model_optimizer = torch.optim.SGD(shadow_model.parameters(), lr=0.1)
    elif 'nn_model' in str(target_model_path):
        target_model = NNClassifier(number_of_features, number_of_classes)
        shadow_model = NNClassifier(number_of_features, number_of_classes)
        shadow_model_optimizer = torch.optim.AdamW(shadow_model.parameters(), lr=0.1)
    else:
        raise ValueError(f"Model path {target_model_path} is not a valid model path!")

    target_model.load_state_dict(remove_model_prefix(torch.load(target_model_path), '_module.'))
    target_model = PyTorchClassifier(target_model, torch.nn.BCELoss(),
                                     input_shape=(number_of_features,), nb_classes=2)
    shadow_model = PyTorchClassifier(shadow_model, torch.nn.BCELoss(), optimizer=shadow_model_optimizer,
                                     input_shape=(number_of_features,), nb_classes=2)

    shadow_models = ShadowModels(shadow_model, num_shadow_models=config.number_of_shadow_models)
    shadow_dataset = shadow_models.generate_shadow_dataset(shadow_model_data['features'],
                                                           shadow_model_data['classes'])
    (member_x, member_y, member_predictions), (non_member_x, non_member_y, non_member_predictions) = shadow_dataset

    shadow_models_evaluation_results = []
    for shadow_model in shadow_models.get_shadow_models():
        prediction_logits = shadow_model.predict(target_model_test_data['features'])
        prediction_classes = np.round(prediction_logits, decimals=0)

        shadow_models_evaluation_results.append({
            "AUROC": roc_auc_score(target_model_test_data['classes'], prediction_logits,
                                   multi_class='ovr', average='macro'),
            "Acc": accuracy_score(target_model_test_data['classes'], prediction_classes),
            "Macro F1 Score": f1_score(target_model_test_data['classes'], prediction_classes, average='macro'),
            "Binary F1 Score": f1_score(target_model_test_data['classes'], prediction_classes, average='binary'),
        })

    # This mia library works only with at least 2 classes
    member_predictions = np.concatenate([1 - member_predictions, member_predictions], axis=1)
    non_member_predictions = np.concatenate([1 - non_member_predictions, non_member_predictions], axis=1)
    attack = MembershipInferenceBlackBox(target_model)
    attack.fit(member_x, member_y, non_member_x, non_member_y, member_predictions, non_member_predictions)

    target_prediction = target_model.predict(target_model_train_data['features'])
    target_prediction = np.concatenate([1 - target_prediction, target_prediction], axis=1)
    membership_inference = attack.infer(target_model_train_data['features'],
                                        target_model_train_data['classes'],
                                        pred=target_prediction)

    target_prediction = target_model.predict(target_model_test_data['features'])
    target_prediction = np.concatenate([1 - target_prediction, target_prediction], axis=1)
    non_membership_inference = attack.infer(target_model_test_data['features'],
                                            target_model_test_data['classes'],
                                            pred=target_prediction)

    predicted_membership = np.concatenate([membership_inference, non_membership_inference])
    true_membership = np.concatenate([np.ones(len(membership_inference)),
                                      np.zeros(len(non_membership_inference))])

    results = {
        "fold_index": fold_index,
        "attack_accuracy": accuracy_score(true_membership, predicted_membership),
        "attack_precision": precision_score(true_membership, predicted_membership),
        "attack_recall": recall_score(true_membership, predicted_membership),
        "shadow_model_scores": shadow_models_evaluation_results,
    }

    with open(RESULTS_PATH / "mia" / (target_model_path.name[:-3] + 'json'), "w") as f:
        json.dump(results, f)


def run_membership_inference_attack(
        dataset_name: str, number_of_features: int, number_of_classes: int, model_paths: list[os.PathLike]
) -> None:
    federated_model_paths = []
    centralised_model_paths = []
    for model_path in model_paths:
        if "_fl" in model_path.name:
            federated_model_paths.append(model_path)
        else:
            centralised_model_paths.append(model_path)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    # Centralised model attack
    print("Attacking centralised models...")
    for fold_index, (target_train_data, target_test_data,
                     shadow_train_data) in DATASET_MANAGER.get_mia_data_folds(dataset_name):
        fold_models = [p for p in model_paths if f"fold={fold_index}" in str(p)]

        target_test_data = next(iter(target_test_data))
        shadow_train_data = next(iter(shadow_train_data))

        def _run_attack_wrapper(model_path: os.PathLike):
            _run_attack(model_path, fold_index,
                        number_of_features, number_of_classes,
                        target_train_data, target_test_data,
                        shadow_train_data)

        with Pool(processes=multiprocessing.cpu_count() // 2) as pool:
            wrap_task(pool, _run_attack_wrapper, fold_models)

    # Federated learning with malicious client attack
    print("Attacking federated models...")
    federated_learning_models = {}
    for model_path in federated_model_paths:
        number_of_clients = int(re.findall(r"clients=\d+", model_path.name)[0].replace("clients=", ""))
        if number_of_clients not in federated_learning_models:
            federated_learning_models[number_of_clients] = []
        federated_learning_models[number_of_clients].append(model_path)

    for number_of_clients, model_paths in federated_learning_models.items():
        for fold_index, (
                target_train_data, target_test_data, shadow_train_data) in DATASET_MANAGER.get_mia_data_folds(
            dataset_name,
            use_federated_data_only=True,
            number_of_clients=number_of_clients,
        ):
            fold_models = [p for p in model_paths if f"fold={fold_index}" in str(p)]

            target_test_data = next(iter(target_test_data))
            shadow_train_data = next(iter(shadow_train_data))

            def _run_attack_wrapper(model_path: os.PathLike):
                _run_attack(model_path, fold_index,
                            number_of_features, number_of_classes,
                            target_train_data, target_test_data,
                            shadow_train_data)

            with Pool(processes=multiprocessing.cpu_count() // 2) as pool:
                wrap_task(pool, _run_attack_wrapper, fold_models)
