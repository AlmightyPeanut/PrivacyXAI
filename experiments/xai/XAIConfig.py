from dataclasses import dataclass, field

from captum._utils.models import SkLearnLasso, SkLearnLinearRegression
from quantus import FaithfulnessEstimate, Monotonicity, Selectivity, Sufficiency, AvgSensitivity, \
    LocalLipschitzEstimate, MaxSensitivity, Complexity, EffectiveComplexity, Sparseness

from experiments.xai.CustomSensitivityN import CustomSensitivityN


@dataclass
class XAIConfig:
    methods: dict = field(default_factory=lambda: {
        'Lime+Lasso': ("Lime", {
            "interpretable_model": SkLearnLasso(alpha=.0)
        }),
        'Lime+LR': ("Lime", {
            "interpretable_model": SkLearnLinearRegression()
        }),
        'Integrated Gradients': ("IntegratedGradients", {}),
        'Feature Permutation': ("FeaturePermutation", {})
    })

    xai_metrics: dict = field(default_factory=lambda: {
        # Faithfulness
        "Faithfulness Estimate": FaithfulnessEstimate(),  # works
        "Monotonicity": Monotonicity(),  # works
        "Selectivity": Selectivity(),  # works
        "SensitivityN": CustomSensitivityN(),  # works
        "Sufficiency": Sufficiency(),  # works

        # Robustness (Takes a long time)
        # "Avg Sensitivity": AvgSensitivity(),  # works
        # "Local Lipschitz Estimate": LocalLipschitzEstimate(),  # works
        # "Max Sensitivity": MaxSensitivity(),  # works

        # Complexity
        "Complexity": Complexity(),  # works
        "Effective Complexity": EffectiveComplexity(),  # TODO: works, but check outputs
        "Sparseness": Sparseness(),  # works

        # Sensitivity (Randomness)
        # "Efficient MPRT": EfficientMPRT(),  # works
        # "MPRT": MPRT(),  # TODO: works, but check outputs
        # "Smooth MPRT": SmoothMPRT(),  # TODO: works, but check outputs

        # Axiomatic
        # "Completeness": Completeness(),  # TODO: works, check outputs
        # "Non-Sensitivity": NonSensitivity(),  # TODO: works, check outputs
    })
