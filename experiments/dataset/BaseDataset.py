import numpy as np
import polars as pl

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self):
        self.data_size = 0
        self.features = None
        self.classes = None

        self.load_data()

    def load_data(self):
        raise NotImplementedError("Needs to be implemented in child class")

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> dict[str, np.array]:
        return self.__getitems__([index])

    def __getitems__(self, index: list[int]) -> dict[str, np.array]:
        sample_selection = pl.col(list(map(str, index)))
        features = self.features.select(sample_selection).transpose().to_numpy().astype(np.float32)
        classes = self.classes.select(sample_selection).transpose().to_numpy().astype(np.int32)  # .squeeze(1)

        samples = {
            'features': features,
            'classes': classes,
        }

        return samples

    def get_number_of_features(self) -> int:
        return self.features.shape[0]

    def get_number_of_classes(self) -> int:
        return self.classes.shape[0]
