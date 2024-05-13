from dataclasses import dataclass, field

SUPPORTED_DATASETS = {'MIMIC', 'Iris', 'MIMICExtract', 'BreastCancer'}


@dataclass
class DatasetConfig:
    datasets: set = field(default_factory=lambda: {'MIMICExtract'})
    batch_size: int = field(default=128)
    test_split: int = 0.2
