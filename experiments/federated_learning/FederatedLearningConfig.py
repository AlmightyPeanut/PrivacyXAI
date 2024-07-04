from dataclasses import dataclass, field


@dataclass
class FederatedLearningConfig:
    batch_size: int = 32
    validation_split: float = 0.1

    fraction_fit: float = .1
    fraction_evaluate: float = .0
    num_rounds: int = 100

    client_training_epochs: int = 5
    client_resources: dict = field(default_factory=lambda: {
        "num_cpus": 20,
        "num_gpus": 1,
    })
