import polars as pl

from experiments.dataset.BaseDataset import BaseDataset
from experiments.utils import DATASET_PATH


class IrisDataset(BaseDataset):
    def load_data(self):
        data = pl.read_csv(DATASET_PATH / 'iris.csv', new_columns=['sepal_length', 'sepal_width', 'petal_length',
                                                                   'petal_width', 'class'])
        data = data.with_columns(
            pl.col('class').rank('dense').cast(pl.Int64) - 1
        )

        self.data_size = len(data)

        new_column_names = list(map(str, range(data.shape[0])))
        self.features = data.select(pl.all().exclude('class')).transpose(column_names=new_column_names)
        self.classes = data.select(pl.col('class')).transpose(column_names=new_column_names)
