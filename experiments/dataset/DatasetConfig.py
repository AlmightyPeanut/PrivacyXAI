from dataclasses import dataclass, field

SUPPORTED_DATASETS = {'MIMIC'}


@dataclass
class DatasetConfig:
    datasets: set = field(default_factory=lambda: {'MIMIC'})
    batch_size: int = field(default=4096)
    test_split: int = 10000
