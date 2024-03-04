import os.path
import polars as pl
import torch

from pathlib import Path
from torch.utils.data import Dataset

MIMIC_DATA_FILE_PATH = Path(__file__).parent.parent.parent / "data" / "MIMIC" / "mimic_data_processed.csv"


class MIMICDataset(Dataset):
    def __init__(self, transform=None):
        data = pl.read_csv(MIMIC_DATA_FILE_PATH).select(pl.all().exclude(
            ['icd_codes', 'subject_id', 'hadm_id', 'nights_of_stay'])).cast(pl.Float64)

        self.data_size = len(data)
        self.features = data.select(pl.all().exclude('nights_of_stay_group')).transpose()
        self.classes = data.select(pl.col('nights_of_stay_group')).transpose()

        self.transform = transform

    def __len__(self):
        return self.data_size

    def __getitem__(self, index: int) -> torch.Tensor:
        features = self.features.select(pl.col(self.features.columns[index])).transpose().to_numpy()
        classes = self.classes.select(pl.col(self.classes.columns[index])).transpose().to_numpy()

        sample = {
            'features': torch.tensor(features),
            'classes': torch.tensor(classes)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitems__(self, index: list[int]) -> torch.Tensor:
        features = self.features.select(pl.col([self.features.columns[i] for i in index])).transpose().to_numpy()
        classes = self.classes.select(pl.col([self.features.columns[i] for i in index])).transpose().to_numpy()

        samples = {
            'features': torch.tensor(features),
            'classes': torch.tensor(classes)
        }

        if self.transform:
            samples = self.transform(samples)

        return samples

