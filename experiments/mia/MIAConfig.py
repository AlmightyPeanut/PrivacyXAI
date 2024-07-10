from dataclasses import dataclass


@dataclass
class MIAConfig:
    number_of_shadow_models: int = 3
