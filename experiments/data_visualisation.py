import re

import numpy as np
import polars as pl
import seaborn as sns
import tqdm
from matplotlib import pyplot as plt

from experiments.utils import DATASET_PATH, RESULTS_PATH

sns.set(font_scale=2)


def main():
    features = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features.csv')
    missing_data_mask_columns = [features.columns[i] for i, data_type in enumerate(features.row(0)) if
                                 data_type == 'mask']
    time_since_measured_columns = [features.columns[i] for i, data_type in enumerate(features.row(0)) if
                                   data_type == 'time_since_measured']

    features = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_features_no_duplicates.csv')
    features = features.drop(missing_data_mask_columns).drop(time_since_measured_columns)

    features = features.with_columns(
        pl.col('*').cast(pl.Float64)
    )

    classes = pl.read_csv(DATASET_PATH / 'MIMIC' / 'mimic_extract_classes.csv')
    classes = classes.select(pl.when(pl.col("los_class") > 0).then(1).otherwise(0).alias('los_class'))

    feature_names = set([re.sub(r'_duplicated_\d+', '', col) for col in features.columns])
    for feature_name in tqdm.tqdm(feature_names):
        # select only data for this feature
        feature_mean_columns = [col for col in features.columns if col.startswith(feature_name)]
        feature_mean_data = features.select(pl.col(feature_mean_columns))

        # make the hours of measurement the new column names
        new_columns = list(range(24))
        number_of_measurements = len(new_columns) // 24
        for i in range(number_of_measurements):
            if i == 0:
                continue

            for j in range(24 * i, len(new_columns)):
                new_columns[j] += 24
        feature_mean_data.columns = [str(col) for col in new_columns]

        feature_mean_data = feature_mean_data.with_row_index().filter((pl.col("index") != 0)).drop("index")

        # join with classes and unpivot the data
        feature_mean_data = feature_mean_data.with_row_index().join(classes.with_row_index(), on='index',
                                                                    how='left').drop('index')
        feature_mean_data = feature_mean_data.melt(id_vars='los_class', variable_name='hour',
                                                   value_name=feature_name).with_columns(
            pl.col('hour').cast(pl.Int16)
        )

        fig, axes = plt.subplots(nrows=2, ncols=number_of_measurements, figsize=(8 * number_of_measurements, 16))
        for i in range(number_of_measurements):
            data = feature_mean_data.filter((0 + 24 * i <= pl.col('hour')) &
                                            (pl.col('hour') < 24 * (i + 1))).with_columns(
                pl.col('hour') - 24 * i
            )

            if number_of_measurements == 1:
                ax_upper = axes[0]
            else:
                ax_upper = axes[0, i]

            line_plot = sns.lineplot(data=data, x='hour', y=feature_name, hue='los_class', ax=ax_upper)

            new_legend_labels = ['Length of stay ≤ 3', 'Length of stay > 3']
            line_plot.legend_.set_title('')
            for old_label, new_label in zip(line_plot.legend_.texts, new_legend_labels):
                old_label.set_text(new_label)

            new_xticks = np.arange(0, 24, 4)

            ax_upper.set_xlim(0, 23)
            ax_upper.set(xticks=new_xticks, xticklabels=new_xticks)
            ax_upper.set_xlabel('')
            ax_upper.set_ylabel('')
            if number_of_measurements > 1:
                ax_upper.set_title(f"Measurement {i + 1}")

            if number_of_measurements == 1:
                ax_lower = axes[1]
            else:
                ax_lower = axes[1, i]

            box_plot = sns.boxplot(data=data, x='hour', y=feature_name, hue='los_class', ax=ax_lower)
            box_plot.legend_.set_title('')

            box_plot.set_title('')
            for old_label, new_label in zip(box_plot.legend_.texts, new_legend_labels):
                old_label.set_text(new_label)
            sns.move_legend(ax_lower, 'upper right')

            ax_lower.set(xticks=new_xticks, xticklabels=new_xticks)
            # ax_lower.set_yscale('symlog')
            ax_lower.set_xlabel('Hour of measurement')
            ax_lower.set_ylabel('')

        fig.suptitle(feature_name)
        plt.tight_layout()
        plt.savefig(RESULTS_PATH / 'feature_plots' / f'{feature_name}.png', bbox_inches='tight')
        # plt.show()
        plt.close(fig)


if __name__ == '__main__':
    main()
