from dataclasses import dataclass


@dataclass
class FederatedLearningConfig:
    batch_size: int = 128
    validation_split: float = 0.1

    # TODO: improve namings
    fraction_fit: float = 1.0
    fraction_evaluate: float = 1.0
    num_rounds: int = 20
