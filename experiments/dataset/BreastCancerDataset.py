import polars as pl

from experiments.dataset.BaseDataset import BaseDataset
from experiments.utils import DATASET_PATH


class BreastCancerDataset(BaseDataset):
    def load_data(self):
        new_columns = ['id', 'diagnosis']
        for i in range(3):
            new_columns.extend([feature_name + str(i) for feature_name in [
                'radius_',
                'texture_',
                'perimeter_',
                'area_',
                'smoothness_',
                'compactness_',
                'concavity_',
                'concave_points_',
                'symmetry_',
                'fractal_dimension_',
            ]])
        data = pl.read_csv(DATASET_PATH / 'wdbc.csv', has_header=False, new_columns=new_columns)

        features = data.select(pl.all().exclude('id', 'diagnosis'))
        classes = data.select(pl.when(pl.col('diagnosis') == 'M').then(1).otherwise(0).alias('malignancy'))

        self.data_size = len(features)

        new_column_names = list(map(str, range(features.shape[0])))
        self.features = features.transpose(column_names=new_column_names)
        self.classes = classes.transpose(column_names=new_column_names)
