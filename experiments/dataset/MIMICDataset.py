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

        self.transform = transform

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> np.array:
        return self.__getitems__([index])

    def __getitems__(self, index: list[int]) -> np.array:
        sample_selection = pl.col(list(map(str, index)))
        features = self.features.select(sample_selection).transpose().to_numpy()
        classes = self.classes.select(sample_selection).transpose().to_numpy().squeeze(1)

        samples = {
            'features': features,
            'classes': classes
        }

        if self.transform:
            samples = self.transform(samples)

        return samples

