import os
import re

import polars as pl

from experiments.dataset.BaseDataset import BaseDataset
from experiments.utils import DATASET_PATH


class MIMICExtractDataset(BaseDataset):

    def load_data(self):
        if not os.path.exists(DATASET_PATH / 'MIMIC' / 'mimic_extract_features_no_duplicates.csv'):
            print("Removing duplicate features from dataset... ", end='')
            self._remove_duplicates_from_data()
            print("Done!")

        features = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features_no_duplicates.csv')
        classes = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_classes.csv')
        classes = classes.select(pl.when(pl.col("los_class") > 0).then(1).otherwise(0))

        self.data_size = len(features)

        new_column_names = list(map(str, range(features.shape[0])))
        self.features = features.transpose(column_names=new_column_names)
        self.classes = classes.transpose(column_names=new_column_names)

    def _remove_duplicates_from_data(self):
        features = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features.csv')

        missing_data_mask_columns = [features.columns[i] for i, data_type in enumerate(features.row(0)) if
                                     data_type == 'mask']
        time_since_measured_columns = [features.columns[i] for i, data_type in enumerate(features.row(0)) if
                                       data_type == 'time_since_measured']
        features = features.with_row_index().filter((pl.col("index") != 0) & (pl.col("index") != 1)).drop("index")
        mean_data = features.drop(missing_data_mask_columns).drop(time_since_measured_columns).with_columns(
            pl.col('*').cast(pl.Float64)
        )

        feature_names = set([re.sub(r'_duplicated_\d+', '', col) for col in features.columns])
        mean_data_split_by_feature, feature_columns_mapping = self._split_data_by_features(mean_data, feature_names)
        duplicate_features = self._find_duplicate_features(mean_data_split_by_feature)
        features_to_drop = self._select_features_to_drop(duplicate_features)
        column_names_to_drop = []
        for feature in features_to_drop:
            column_names_to_drop.extend(feature_columns_mapping[feature])
        mean_data = mean_data.drop(column_names_to_drop)

        mean_data.write_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features_no_duplicates.csv')

    @staticmethod
    def _split_data_by_features(mean_data: pl.DataFrame, feature_names: set[str]) -> (dict[str, pl.DataFrame],
                                                                                      dict[str, list[str]]):
        mean_data_split_by_feature = dict()
        feature_columns_mapping = dict()
        for feature_name in feature_names:
            # select only data for this feature
            feature_mean_columns = [col for col in mean_data.columns if col.startswith(feature_name)]
            feature_mean_data = mean_data.select(pl.col(feature_mean_columns))

            new_column_names = list(str(i) for i in range(24))

            number_of_measurements = len(feature_mean_data.columns) // 24
            if number_of_measurements == 1:
                if feature_name in mean_data_split_by_feature:
                    raise RuntimeError(f"{feature_name} already exists!")

                feature_columns_mapping[feature_name] = feature_mean_data.columns
                feature_mean_data.columns = new_column_names
                mean_data_split_by_feature[feature_name] = feature_mean_data
                continue

            for i in range(number_of_measurements):
                feature_key = feature_name + '_' + str(i)
                if feature_key in mean_data_split_by_feature:
                    raise RuntimeError(f"{feature_key} already exists!")

                feature_mean_data_values = feature_mean_data.select(pl.col(
                    feature_mean_columns[i * 24:(i + 1) * 24]
                ))
                feature_columns_mapping[feature_key] = feature_mean_data_values.columns
                feature_mean_data_values.columns = new_column_names
                mean_data_split_by_feature[feature_key] = feature_mean_data_values

        return mean_data_split_by_feature, feature_columns_mapping

    @staticmethod
    def _find_duplicate_features(mean_data_split_by_feature: dict[str, pl.DataFrame]) -> list[list[str]]:
        duplicate_features = dict()
        for i, (feature, values) in enumerate(mean_data_split_by_feature.items()):
            for j, (other_feature, other_values) in enumerate(mean_data_split_by_feature.items()):
                if j <= i:
                    continue

                if values.equals(other_values):
                    if other_feature in duplicate_features:
                        duplicate_features[other_feature].append(feature)
                        continue

                    if feature not in duplicate_features:
                        duplicate_features[feature] = [feature]
                    duplicate_features[feature].append(other_feature)

        return list(duplicate_features.values())

    @staticmethod
    def _select_features_to_drop(duplicate_features: list[list[str]]):
        features_to_drop = []
        for duplicate_feature_list in duplicate_features:
            found_descriptive_feature_name = False
            for feature_index, duplicate_feature in enumerate(duplicate_feature_list):
                if not re.match(r"^.*_\d+$", duplicate_feature):
                    found_descriptive_feature_name = True
                    features_to_drop.extend(duplicate_feature_list[:feature_index])
                    if feature_index + 1 < len(duplicate_feature_list):
                        features_to_drop.extend(duplicate_feature_list[feature_index + 1:])
                    break

            if not found_descriptive_feature_name:
                features_to_drop.extend(duplicate_feature_list[:-1])

        return features_to_drop
