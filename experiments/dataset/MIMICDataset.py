import numpy as np
import polars as pl

from pathlib import Path
from torch.utils.data import Dataset

MIMIC_DATA_FILE_PATH = Path(__file__).parent.parent.parent / "data" / "MIMIC" / "mimic_data_processed.csv"


class MIMICDataset(Dataset):
    def __init__(self, transform=None):
        data = pl.read_csv(MIMIC_DATA_FILE_PATH).select(pl.all().exclude(
            ['icd_codes', 'subject_id', 'hadm_id', 'nights_of_stay'])).cast(pl.Float64)

        self.data_size = len(data)

        new_column_names = list(map(str, range(data.shape[0])))
        self.features = data.select(pl.all().exclude('nights_of_stay_group')).transpose(column_names=new_column_names)
        self.classes = data.select(pl.col('nights_of_stay_group')).transpose(column_names=new_column_names)
        self.train_test_indicator = None

        self.transform = transform

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> np.array:
        return self.__getitems__([index])

    def __getitems__(self, index: list[int]) -> np.array:
        sample_selection = pl.col(list(map(str, index)))
        features = self.features.select(sample_selection).transpose().to_numpy()
        features = (features - features.min()) / (features.max() - features.min())
        classes = self.classes.select(sample_selection).transpose().to_numpy().astype(np.int64).squeeze(1)
        train_test_indicator = self.train_test_indicator.select(sample_selection).transpose().to_numpy().astype(
            np.int64).squeeze(1)

        samples = {
            'features': features,
            'classes': classes,
            'train_test_indicator': train_test_indicator
        }

        if self.transform:
            samples = self.transform(samples)

        return samples

    def add_train_test_indicator(self, test_inidices: list[int]) -> None:
        new_column_names = list(map(str, range(self.data_size)))
        self.train_test_indicator = pl.DataFrame({
            'is_train_sample': [0 if sample_idx in test_inidices else 1 for sample_idx in range(self.data_size)]
        }).transpose(column_names=new_column_names)
