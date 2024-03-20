from dataclasses import dataclass


@dataclass
class FederatedLearningConfig:
    number_of_clients: int = 2
    batch_size: int = 4096  # 256
    validation_ratio: float = .1
    privatise_models: bool = True
    # TODO: improve namings
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.5
    num_rounds: int = 1
