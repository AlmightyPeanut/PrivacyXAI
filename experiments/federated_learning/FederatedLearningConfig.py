from dataclasses import dataclass


@dataclass
class FederatedLearningConfig:
    number_of_clients: int = 5
    batch_size: int = 128
    privatise_models: bool = True
    validation_split: float = 0.1

    # TODO: improve namings
    fraction_fit: float = 0.5
    fraction_evaluate: float = 0.5
    num_rounds: int = 1
