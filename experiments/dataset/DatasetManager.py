from torch.utils.data import DataLoader, Dataset, random_split

from .MIMICDataset import MIMICDataset
from .DatasetConfig import DatasetConfig, SUPPORTED_DATASETS


class DatasetManager:
    def __init__(self, config: DatasetConfig = DatasetConfig()):
        self.config = config
        self.collate_fn = lambda x: x

    def load_datasets(self) -> dict[str, (DataLoader, DataLoader)]:
        datasets = {}

        for dataset_string in self.config.datasets.union(SUPPORTED_DATASETS):
            if dataset_string == 'MIMIC':
                all_data = MIMICDataset()
                train_data, test_data = random_split(all_data,
                                                     [len(all_data) - self.config.test_split, self.config.test_split])
                datasets['MIMIC'] = (DataLoader(train_data, batch_size=self.config.batch_size, collate_fn=self.collate_fn),
                                     DataLoader(test_data, batch_size=len(test_data), collate_fn=self.collate_fn))

        return datasets
