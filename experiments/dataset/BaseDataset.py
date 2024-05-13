import numpy as np
import polars as pl

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, transform=None):
        self.data_size = 0
        self.features = None
        self.classes = None
        self.train_test_indicator = None
        self.transform = transform

        self.load_data()

    def load_data(self):
        raise NotImplementedError("Needs to be implemented in child class")

    def __len__(self) -> int:
        return self.data_size

    def __getitem__(self, index: int) -> np.array:
        return self.__getitems__([index])

    def __getitems__(self, index: list[int]) -> np.array:
        sample_selection = pl.col(list(map(str, index)))
        features = self.features.select(sample_selection).transpose().to_numpy().astype(np.float64)
        classes = self.classes.select(sample_selection).transpose().to_numpy().astype(np.int64)  # .squeeze(1)

        train_test_indicator = None
        if self.train_test_indicator is not None:
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

        is_train_sample = np.ones(self.data_size, dtype=np.int64)
        is_train_sample[test_inidices] = 0
        self.train_test_indicator = pl.DataFrame({
            'is_train_sample': is_train_sample
        }).transpose(column_names=new_column_names)

    def get_number_of_features(self) -> int:
        return self.features.shape[0]

    def get_number_of_classes(self) -> int:
        return self.classes.shape[0]
