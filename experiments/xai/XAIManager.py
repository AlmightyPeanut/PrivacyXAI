import os

import torch
from torch.multiprocessing import Pool

try:
    torch.multiprocessing.set_start_method('spawn')
except RuntimeError:
    pass

import numpy as np
import pandas as pd
import quantus
import torch
from torch.utils.data import DataLoader

from experiments.xai.XAIConfig import XAIConfig
from experiments.model.LRClassifier import LRClassifier
from experiments.model.NNClassifier import NNClassifier
from experiments.utils import RESULTS_PATH, remove_model_prefix, wrap_task

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
elif torch.backends.mps.is_available():
    TORCH_DEVICE = "mps"
else:
    TORCH_DEVICE = "cpu"

torch.manual_seed(42)


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
    if torch.cuda.is_available():
        gpu_id = np.random.randint(0, torch.cuda.device_count())
        torch.cuda.set_device(gpu_id)

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
