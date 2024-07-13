import multiprocessing
import torch

from dataclasses import dataclass, field
from captum._utils.models import SkLearnLasso, SkLearnLinearRegression

from quantus import (
    FaithfulnessEstimate,
    Monotonicity,
    Selectivity,
    Sufficiency,
    AvgSensitivity,
    LocalLipschitzEstimate,
    MaxSensitivity,
    Complexity,
    EffectiveComplexity,
    Sparseness,
    EfficientMPRT,
    MPRT,
    SmoothMPRT,
    Completeness,
    NonSensitivity,
)

from experiments.xai.CustomSensitivityN import CustomSensitivityN

if torch.cuda.is_available():
    TORCH_DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    TORCH_DEVICE = torch.device("mps")
else:
    TORCH_DEVICE = torch.device("cpu")


@dataclass
class XAIConfig:
    processes: int = multiprocessing.cpu_count() // 2
    number_of_xai_test_samples: int = 5
    methods: dict = field(default_factory=lambda: {
        "Lime": {
            "method": "Lime",
            "xai_lib_kwargs": {
                "interpretable_model": SkLearnLasso(alpha=.0),
            },
            "xai_lib": 'captum',
            "device": TORCH_DEVICE,
        },
        # "Lime+LR": {
        #     "method": "Lime",
        #     "xai_lib_kwargs": {
        #         "interpretable_model": SkLearnLinearRegression()
        #     },
        #     "xai_lib": 'captum',
        #     "device": TORCH_DEVICE,
        # },
        'IntegratedGradients': {
            "method": "IntegratedGradients",
            "xai_lib": 'captum',
            "device": TORCH_DEVICE,
        },
        'FeaturePermutation': {
            "method": "FeaturePermutation",
            "xai_lib": 'captum',
            "device": TORCH_DEVICE,
        }
    })

    xai_metrics: dict = field(default_factory=lambda: {
        # Faithfulness
        "Faithfulness Estimate": FaithfulnessEstimate(
            perturb_baseline='mean',
        ),
        "Monotonicity": Monotonicity(
            perturb_baseline='mean',
        ),
        "SensitivityN": CustomSensitivityN(
            perturb_baseline='mean',
        ),
        "Sufficiency": Sufficiency(),

        # Robustness (Takes a long time)
        "Avg Sensitivity": AvgSensitivity(
            nr_samples=100,
            display_progressbar=True,
        ),
        "Local Lipschitz Estimate": LocalLipschitzEstimate(
            nr_samples=100,
            display_progressbar=True,
        ),
        "Max Sensitivity": MaxSensitivity(
            nr_samples=100,
            display_progressbar=True,
        ),

        # Complexity
        "Complexity": Complexity(),
        "Effective Complexity": EffectiveComplexity(),
        "Sparseness": Sparseness(),

        # Sensitivity (Randomness)
        "Efficient MPRT": EfficientMPRT(),
        "MPRT": MPRT(
            return_aggregate=True,
            return_average_correlation=True,
        ),
        "Smooth MPRT": SmoothMPRT(
            nr_samples=100,
            return_aggregate=True,
            return_average_correlation=True,
        ),

        # Axiomatic
        "Completeness": Completeness(
            perturb_baseline='mean',
        ),
        "Non-Sensitivity": NonSensitivity(
            perturb_baseline='mean',
        ),
    })
