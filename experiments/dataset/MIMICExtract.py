import polars as pl

from experiments.dataset.BaseDataset import BaseDataset
from experiments.utils import DATASET_PATH


class MIMICExtractDataset(BaseDataset):

    def load_data(self):
        features = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features.csv')
        mask_columns_indexes = [i for i, data_type in enumerate(features.row(0)) if data_type == 'mask' or data_type == 'time_since_measured']
        mask_columns_names = [features.columns[i] for i in mask_columns_indexes]
        features = features.drop(mask_columns_names)

        features = features.with_row_count("row_index").filter((pl.col("row_index") != 0)
                                                               & (pl.col("row_index") != 1)).drop("row_index")
        classes = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_classes.csv')
        classes = classes.select(pl.when(pl.col("los_class") > 0).then(1).otherwise(0))

        self.data_size = len(features)

        new_column_names = list(map(str, range(features.shape[0])))
        self.features = features.transpose(column_names=new_column_names)
        self.classes = classes.transpose(column_names=new_column_names)
