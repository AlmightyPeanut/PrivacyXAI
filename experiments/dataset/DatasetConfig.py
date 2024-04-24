from dataclasses import dataclass, field

SUPPORTED_DATASETS = {'MIMIC'}


@dataclass
class DatasetConfig:
    datasets: set = field(default_factory=lambda: {'MIMIC'})
    batch_size: int = field(default=4096)
    test_split: int = 1000
    records_to_attack_split: int = 500
    fl_validation_ratio: float = 0.1
    number_of_partitions: int = 5
