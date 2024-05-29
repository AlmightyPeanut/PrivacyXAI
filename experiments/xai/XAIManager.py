import json
import os
import time
import warnings

import numpy as np
import quantus
import torch
from torch.utils.data import DataLoader, Dataset

from .XAIConfig import XAIConfig
from ..model.LRClassifier import LRClassifier
from ..model.NNClassifier import NNClassifier
from ..utils import RESULTS_PATH, PRINT_WIDTH

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
else:
    TORCH_DEVICE = torch.device("cpu")

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


class XAIManager:
    def __init__(self, config: XAIConfig = XAIConfig()):
        self.config = config

    def _evaluate_model_explanations(self, data: Dataset, model: torch.nn.Module,
                                     number_of_classes: int) -> dict[str, dict[str, float]]:
        eval_scores = {}
        target_index_to_explain = np.repeat(np.arange(number_of_classes), len(data['classes']), axis=0)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)

            for xai_method_readable_name, (xai_method_name, xai_lib_kwargs) in self.config.methods.items():
                print(f"Evaluating {xai_method_readable_name}...")
                explain_function_kwargs = {
                    "method": xai_method_name,
                    "xai_lib_kwargs": xai_lib_kwargs,
                    "device": TORCH_DEVICE,
                }

                attributions = quantus.explain(model, data['features'], target_index_to_explain,
                                               **explain_function_kwargs)

                xai_method_scores = {}
                for xai_metric_name, xai_metric in self.config.xai_metrics.items():
                    print(f"Starting {xai_metric_name} evaluation...", end='')
                    pre_evaluation_time = time.time()
                    xai_metric_results = xai_metric(
                        model, data['features'], target_index_to_explain, attributions,
                        explain_func=quantus.explain, explain_func_kwargs=explain_function_kwargs,
                        device=TORCH_DEVICE,
                    )
                    print(f" {time.time() - pre_evaluation_time:.2f}s")

                    if xai_metric_name in ["MPRT", "Smooth MPRT"]:
                        xai_metric_results = xai_metric_results['output_layer']
                    elif isinstance(xai_metric_results[0], np.ndarray):
                        xai_metric_results = [np.average(x) for x in xai_metric_results]
                    xai_method_scores[xai_metric_name] = np.average(np.squeeze(xai_metric_results))
                eval_scores[xai_method_readable_name] = xai_method_scores

        return eval_scores

    def evaluate_explanations(self, data: DataLoader, number_of_features: int, number_of_classes: int,
                              model_paths: list[os.PathLike]) -> None:
        data = next(iter(data))

        done_models = [p for p in os.listdir(RESULTS_PATH / 'xai') if p.endswith('.json')]

        for model_file in model_paths:
            model_xai_file = os.path.basename(model_file)[:-3] + 'json'
            if model_xai_file in done_models:
                continue

            if not os.path.exists(model_file):
                raise ValueError(f"Model path {model_file} does not exist!")

            if 'lr_model' in str(model_file):
                model = LRClassifier(number_of_features, number_of_classes)
                continue
            elif 'nn_model' in str(model_file):
                model = NNClassifier(number_of_features, number_of_classes)
            else:
                raise ValueError(f"Model path {model_file} is not a valid model path!")
            model.load_state_dict(remove_model_prefix(torch.load(model_file), '_module.'))

            print(f"Evaluating model {model_file}...")
            eval_scores = self._evaluate_model_explanations(data, model, number_of_classes)

            with open(RESULTS_PATH / 'xai' / model_xai_file, 'w') as f:
                json.dump(eval_scores, f)

            print(f" Model name: {str(model_file).split('/')[-1][:-3]} ".center(PRINT_WIDTH, '-'))
            for metric_name, xai_metric_scores in eval_scores.items():
                print(f"{metric_name}".center(PRINT_WIDTH, '_'))
                for meta_metric_name, meta_metric_score in xai_metric_scores.items():
                    print(f"{meta_metric_name}: {meta_metric_score}, ", end='')
                print()
            print()
            print()
