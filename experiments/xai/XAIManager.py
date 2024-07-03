import marshal
import os
import pickle
import torch
import types
from functools import partial
from torch.multiprocessing import Pool

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import quantus
import torch
import tqdm
from torch.utils.data import DataLoader

from .XAIConfig import XAIConfig
from ..model.LRClassifier import LRClassifier
from ..model.NNClassifier import NNClassifier
from ..utils import RESULTS_PATH

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
else:
    TORCH_DEVICE = "cpu"

torch.manual_seed(42)


def remove_model_prefix(state_dict, prefix: str):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]  # remove prefix
        else:
            new_key = k
        new_state_dict[new_key] = v
    return new_state_dict


def evaluate_model_explanations(data: np.array, model: torch.nn.Module, xai_metrics: dict, xai_methods: dict,
                                number_of_classes: int) -> pd.DataFrame:
    target_index_to_explain = np.repeat(np.arange(number_of_classes), len(data), axis=0)

    eval_scores = quantus.evaluate(
        metrics=xai_metrics,
        xai_methods=xai_methods,
        model=model,
        x_batch=data,
        y_batch=target_index_to_explain,
        agg_func=np.mean,
        verbose=True,
        return_as_df=True,
        explain_func_kwargs={},
        call_kwargs={'0': {'device': TORCH_DEVICE}},
    )

    return eval_scores


def xai_task(data: np.array, model_file: os.PathLike,
             number_of_features: int, number_of_classes: int):
    if not os.path.exists(model_file):
        raise ValueError(f"Model path {model_file} does not exist!")

    if 'lr_model' in str(model_file):
        model = LRClassifier(number_of_features, number_of_classes)
    elif 'nn_model' in str(model_file):
        model = NNClassifier(number_of_features, number_of_classes)
    else:
        raise ValueError(f"Model path {model_file} is not a valid model path!")
    model.load_state_dict(remove_model_prefix(torch.load(model_file), '_module.'))

    print(f"Evaluating model {model_file}...")
    config = XAIConfig()
    eval_scores = evaluate_model_explanations(data, model, config.xai_metrics, config.methods, number_of_classes)

    model_xai_file = os.path.basename(model_file)[:-3] + 'csv'
    eval_scores.to_csv(RESULTS_PATH / 'xai' / model_xai_file)


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


class XAIManager:
    def __init__(self, config: XAIConfig = XAIConfig()):
        self.config = config

    def evaluate_explanations(self, data: DataLoader, number_of_features: int, number_of_classes: int,
                              model_paths: list[os.PathLike]) -> None:
        data = next(iter(data))
        xai_test_samples = data['features']
        if self.config.number_of_xai_test_samples > 0:
            xai_test_samples = xai_test_samples[np.random.choice(len(data['features']),
                                                                 self.config.number_of_xai_test_samples, replace=False)]

        def task_wrapper(model_path: os.PathLike) -> None:
            xai_task(data=xai_test_samples, model_file=model_path,
                     number_of_features=number_of_features, number_of_classes=number_of_classes)

        with Pool(self.config.processes) as p:
            wrap_task(p, task_wrapper, model_paths)
