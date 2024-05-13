from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    target_models: set = field(default_factory=lambda: {'LR', 'NN'})
    number_of_epochs: int = 20

    dp_target_epsilon: float = 0.5
    dp_target_delta: float = 0.01
    dp_max_grad_norm: float = 1.0
