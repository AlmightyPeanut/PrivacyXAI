from torch.utils.data import DataLoader

from .MIMICDataset import MIMICDataset
from .DatasetConfig import DatasetConfig, SUPPORTED_DATASETS


class DatasetManager:
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        self.config = config

    def load_datasets(self) -> dict[str, DataLoader]:
        datasets = {}

        for dataset_string in self.config.datasets.union(SUPPORTED_DATASETS):
            if dataset_string == 'MIMIC':
                datasets['MIMIC'] = DataLoader(MIMICDataset(), batch_size=self.config.batch_size,
                                               shuffle=self.config.shuffle, num_workers=self.config.num_workers,
                                               collate_fn=lambda x: x)

        return datasets
