from dataclasses import dataclass, field

SUPPORTED_DATASETS = {'MIMIC'}


@dataclass
class DatasetConfig:
    datasets: set = field(default_factory=lambda: {'MIMIC'})
    batch_size: int = field(default=64)
    shuffle: bool = field(default=True)
    num_workers: int = field(default=0)
