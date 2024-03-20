from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    target_models: set = field(default_factory=lambda: {'NN', 'LR'})
    number_of_epochs: int = 1

    # TODO: get from dataset
    number_of_features: int = 1000 + 6  # vitals + icd code vector size
    number_of_classes: int = 3

    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
