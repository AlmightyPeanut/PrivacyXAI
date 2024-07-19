import numpy as np
from torch.utils.data import Dataset

from experiments.dataset.BaseDataset import BaseDataset


class NormalisedDataset(Dataset):
    def __init__(self, dataset: BaseDataset, standard_deviation: float, mean: float):
        self.dataset = dataset
        self.standard_deviation = standard_deviation
        self.mean = mean

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, np.array]:
        return self.__getitems__([index])

    def __getitems__(self, index: list[int]) -> dict[str, np.array]:
        samples = self.dataset.__getitems__(index)

        samples["features"] = (samples["features"] - self.mean) / self.standard_deviation

        return samples

    def get_number_of_features(self) -> int:
        return self.dataset.features.shape[0]

    def get_number_of_classes(self) -> int:
        return self.dataset.classes.shape[0]
