import polars as pl

from .BaseDataset import BaseDataset
from experiments.utils import DATASET_PATH

MIMIC_DATA_FILE_PATH = DATASET_PATH / "MIMIC" / "mimic_data_processed.csv"


class MIMICDataset(BaseDataset):
    def load_data(self):
        data = pl.read_csv(MIMIC_DATA_FILE_PATH).select(pl.all().exclude(
            ['icd_codes', 'subject_id', 'hadm_id', 'nights_of_stay'])).cast(pl.Float64)

        self.data_size = len(data)

        new_column_names = list(map(str, range(data.shape[0])))
        self.features = data.select(pl.all().exclude('nights_of_stay_group')).transpose(column_names=new_column_names)
        self.classes = data.select(pl.col('nights_of_stay_group')).transpose(column_names=new_column_names)
