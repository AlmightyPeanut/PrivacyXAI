from dataclasses import dataclass, field


@dataclass
class FederatedLearningConfig:
    batch_size: int = 128
    validation_split: float = 0.1

    # TODO: improve namings
    fraction_fit: float = 1.0
    fraction_evaluate: float = .0
    num_rounds: int = 1

    client_training_epochs: int = 1
    client_resources: dict = field(default_factory=lambda: {
        "num_cpus": 2,
        "num_gpus": 1,
    })
