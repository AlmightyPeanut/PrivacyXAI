import os
import time
import warnings

import numpy as np
import quantus
import torch
from captum._utils.models.linear_model import SkLearnLasso, SkLearnLinearRegression
from quantus import Infidelity, SensitivityN, Continuity, Consistency, Monotonicity, FaithfulnessCorrelation, \
    FaithfulnessEstimate, Selectivity, Sufficiency, LocalLipschitzEstimate, MaxSensitivity, AvgSensitivity, \
    RelativeInputStability, RelativeRepresentationStability, RelativeOutputStability, Sparseness, Complexity, \
    EffectiveComplexity, MPRT, SmoothMPRT, EfficientMPRT, RandomLogit, Completeness, NonSensitivity, InputInvariance
from torch.utils.data import DataLoader, Dataset

from .CustomSensitivityN import CustomSensitivityN
from .XAIConfig import XAIConfig
from experiments.model.ModelManager import ModelManager
from experiments.utils.Singleton import Singleton

if torch.cuda.is_available():
    TORCH_DEVICE = "cuda"
else:
    TORCH_DEVICE = "cpu"


# ####### XAI evaluation on MIMIC ########
# ------------ Model name: LR ------------
# _______________Lime+Lasso_______________
# SensitivityN: -0.3611892645402629, Monotonicity: 1.0, Faithfulness Estimate: 0.011310260311981984,
# Selectivity: 0.3799383535015743, Sufficiency: 0.8, Local Lipschitz Estimate: 1.0100286975338635,
# Max Sensitivity: 1.8574914336204529, Avg Sensitivity: 1.4167297824025153, Sparseness: 0.9299121255977265,
# Complexity: 4.52184230100424, Effective Complexity: 438.1, Efficient MPRT: 0.05137309989443195, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# ________________Lime+LR_________________
# SensitivityN: -0.2695467317827918, Monotonicity: 1.0, Faithfulness Estimate: 0.0481793226879911,
# Selectivity: 0.3770622215211735, Sufficiency: 0.8, Local Lipschitz Estimate: 1.2205589454869834,
# Max Sensitivity: 5.677576386928559, Avg Sensitivity: 2.075033786714077, Sparseness: 0.4702439045483902,
# Complexity: 6.474600364016899, Effective Complexity: 1005.5, Efficient MPRT: 0.04487747340430174, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# __________Integrated Gradients__________
# SensitivityN: 0.3027826962740103, Monotonicity: 1.0, Faithfulness Estimate: 0.9897318767322888,
# Selectivity: 0.3635332144116622, Sufficiency: 0.9, Local Lipschitz Estimate: 0.1961006643364242,
# Max Sensitivity: 0.2998253762722015, Avg Sensitivity: 0.26905462363362315, Sparseness: 0.4393803963870867,
# Complexity: 6.555845873926577, Effective Complexity: 1005.9, Efficient MPRT: 0.28980444967672525, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# __________Feature Permutation___________
# SensitivityN: -0.05595016632707009, Monotonicity: 1.0, Faithfulness Estimate: 0.07745178315287024,
# Selectivity: 0.37917717102336035, Sufficiency: 0.0, Local Lipschitz Estimate: 0.8187669675206972,
# Max Sensitivity: 3.5801393389701843, Avg Sensitivity: 2.746741015553474, Sparseness: 0.6910534539316164,
# Complexity: 5.863430710693797, Effective Complexity: 880.4, Efficient MPRT: 0.4508112447586874, Completeness: 0.0,
# Non-Sensitivity: 1.8,
#
#
# ------------ Model name: NN ------------
# _______________Lime+Lasso_______________
# SensitivityN: 0.13513386920727902, Monotonicity: 1.0, Faithfulness Estimate: 0.029358131219637096,
# Selectivity: 0.3854037962277842, Sufficiency: 0.6, Local Lipschitz Estimate: 0.9737955574089352,
# Max Sensitivity: 1.8511518955230712, Avg Sensitivity: 1.4087294870316982, Sparseness: 0.9232110306287499,
# Complexity: 4.622525700751507, Effective Complexity: 443.4, Efficient MPRT: -0.02683284897032139, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# ________________Lime+LR_________________
# SensitivityN: -0.11171697496534089, Monotonicity: 1.0, Faithfulness Estimate: 0.013071597034386411,
# Selectivity: 0.3854258186845886, Sufficiency: 0.9, Local Lipschitz Estimate: 1.0643884692701027,
# Max Sensitivity: 20.662490487098694, Avg Sensitivity: 5.90923695486784, Sparseness: 0.4812045251746803,
# Complexity: 6.443160329237739, Effective Complexity: 1003.3, Efficient MPRT: 0.06004557316768681, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# __________Integrated Gradients__________
# SensitivityN: 0.005839357938588821, Monotonicity: 1.0, Faithfulness Estimate: 0.7856623678443654,
# Selectivity: 0.3848512887068757, Sufficiency: 0.9, Local Lipschitz Estimate: 0.6904498622027472,
# Max Sensitivity: 0.31359263956546785, Avg Sensitivity: 0.29484805805981157, Sparseness: 0.42066456416734443,
# Complexity: 6.618691286037512, Effective Complexity: 1005.9, Efficient MPRT: 0.07927579931778886, Completeness: 0.0,
# Non-Sensitivity: 2.0,
# __________Feature Permutation___________
# SensitivityN: -0.2406966031301486, Monotonicity: 1.0, Faithfulness Estimate: 0.047614304667342455,
# Selectivity: 0.3855062245293849, Sufficiency: 0.0, Local Lipschitz Estimate: 1.4062796838019462,
# Max Sensitivity: 4.000305128097534, Avg Sensitivity: 3.5117564166784283, Sparseness: 0.6582727801385098,
# Complexity: 6.090985179980433, Effective Complexity: 838.3, Efficient MPRT: 0.07047596012392246, Completeness: 0.0,
# Non-Sensitivity: 2.0,


class XAIManager(metaclass=Singleton):
    def __init__(self, config: XAIConfig = XAIConfig()):
        self.config = config
        self.xai_metrics = {
            # Faithfulness
            "Faithfulness Estimate": FaithfulnessEstimate(),  # works
            "Monotonicity": Monotonicity(),  # works
            "SensitivityN": CustomSensitivityN(),  # works
            "Selectivity": Selectivity(),  # works
            "Sufficiency": Sufficiency(),  # works

            # Robustness (Takes a long time)
            # "Local Lipschitz Estimate": LocalLipschitzEstimate(),  # works
            # "Max Sensitivity": MaxSensitivity(),  # works
            # "Avg Sensitivity": AvgSensitivity(),  # works

            # Complexity
            "Sparseness": Sparseness(),  # works
            "Complexity": Complexity(),  # works
            "Effective Complexity": EffectiveComplexity(),  # TODO: works, but check outputs

            # Sensitivity (Randomness)
            "MPRT": MPRT(),  # TODO: works, but check outputs
            "Smooth MPRT": SmoothMPRT(),  # TODO: works, but check outputs
            "Efficient MPRT": EfficientMPRT(),  # works

            # Axiomatic
            "Completeness": Completeness(),  # TODO: works, check outputs
            "Non-Sensitivity": NonSensitivity(),  # TODO: works, check outputs
        }

    def _evaluate_model_explanations(self, data: Dataset, model: torch.nn.Module) -> dict[str, dict[str, float]]:
        eval_scores = {}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for xai_method_readable_name, (xai_method_name, xai_lib_kwargs) in {
                'Lime+Lasso': ("Lime", {
                    "interpretable_model": SkLearnLasso(alpha=.0)
                }),
                'Lime+LR': ("Lime", {
                    "interpretable_model": SkLearnLinearRegression()
                }),
                'Integrated Gradients': ("IntegratedGradients", {}),
                'Feature Permutation': ("FeaturePermutation", {})
            }.items():
                print(f"Evaluating {xai_method_readable_name}...")

                explain_function_kwargs = {
                    "method": xai_method_name,
                    "xai_lib_kwargs": xai_lib_kwargs,
                    "device": TORCH_DEVICE,
                }

                attributions = quantus.explain(model, data['features'], data['classes'], **explain_function_kwargs)

                xai_method_scores = {}
                for xai_metric_name, xai_metric in self.xai_metrics.items():
                    print(f"Starting {xai_metric_name} evaluation...", end='')
                    pre_evaluation_time = time.time()
                    xai_metric_results = xai_metric(
                        model, data['features'], data['classes'], attributions,
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

    def evaluate_explanations(self, data: DataLoader, model_folder: os.PathLike) -> (
            dict[str, dict[str, dict[str, float]]]):
        model_manager = ModelManager()
        model_manager.load_models(model_folder)

        data = next(iter(data))

        model_xai_scores = {}
        if 'LR' in model_manager.config.target_models:
            model_xai_scores['LR'] = self._evaluate_model_explanations(data, model_manager.lr_model)

        if 'SVC' in model_manager.config.target_models:
            raise NotImplementedError("SVC is not supported for XAI evaluation")

        if 'NN' in model_manager.config.target_models:
            model_xai_scores['NN'] = self._evaluate_model_explanations(data, model_manager.nn_model)

        return model_xai_scores


XAI_MANAGER = XAIManager()
