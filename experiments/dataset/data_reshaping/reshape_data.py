from functools import partial

import h5py
from icdmappings import Mapper
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from sklearn.metrics.pairwise import rbf_kernel

DATA_DIR_ROOT = Path(__file__).parent.parent.parent.parent / 'data' / 'MIMIC'
DATA_DIR_ORIGINAL = DATA_DIR_ROOT / 'original'


def reshape_omr_data() -> pl.DataFrame:
    omr_data = pl.read_csv(DATA_DIR_ORIGINAL.joinpath('omr.csv'))
    omr_data = omr_data.pivot(index='subject_id', columns='result_name', values='result_value',
                              aggregate_function='max')
    omr_data = omr_data.select(pl.col(
        'subject_id',
        'BMI',
        'BMI (kg/m2)',
        'Blood Pressure',
        'Blood Pressure Sitting',
        'Blood Pressure Lying',
        'Blood Pressure Standing',
        'Blood Pressure Standing (1 min)',
        'Blood Pressure Standing (3 mins)',
        'Weight',
        'Weight (Lbs)',
        'Height',
        'Height (Inches)',
        'eGFR',
    ))

    with pl.Config(tbl_cols=-1, tbl_width_chars=1000):
        print(omr_data.describe())
    breakpoint()
    # TODO: data is shit

    omr_data = omr_data.with_columns(
        pl.when(pl.col('BMI (kg/m2)').is_null()).then(pl.col('BMI')).otherwise(pl.col('BMI (kg/m2)')).alias('BMI').cast(
            pl.Float64),
        pl.coalesce([
            'Blood Pressure',
            'Blood Pressure Sitting',
            'Blood Pressure Lying',
            'Blood Pressure Standing',
            'Blood Pressure Standing (1 min)',
            'Blood Pressure Standing (3 mins)',
        ]).str.split('/').alias('BP'),
        pl.when(pl.col('Height').is_null()).then(pl.col('Height (Inches)').cast(pl.Float64) * 2.54)
        .otherwise(pl.col('Height')).alias('Height').cast(pl.Float64),
        pl.when(pl.col('Weight').is_null()).then(pl.col('Weight (Lbs)').cast(pl.Float64) * 0.4535924)
        .otherwise(pl.col('Weight')).alias('Weight').cast(pl.Float64),
        pl.col('eGFR').str.replace(r'[>|\s]+', '').alias('eGFR').cast(pl.Float64)
    ).drop([
        'BMI (kg/m2)',
        'Blood Pressure',
        'Blood Pressure Sitting',
        'Blood Pressure Lying',
        'Blood Pressure Standing',
        'Blood Pressure Standing (1 min)',
        'Blood Pressure Standing (3 mins)',
        'Height (Inches)',
        'Weight (Lbs)',
    ]).with_columns(
        pl.col('BP').list.first().alias('BP_sys').cast(pl.Float64),
        pl.col('BP').list.last().alias('BP_dia').cast(pl.Float64),
    ).drop('BP')

    print(omr_data.describe())
    omr_data = omr_data.select(pl.col('subject_id'),
                               (pl.all().exclude('subject_id') - pl.all().exclude('subject_id').mean())
                               / pl.all().exclude('subject_id').std())
    print(omr_data.describe())

    return omr_data


def reshape_diagnoses_data() -> pl.DataFrame:
    diagnoses_data = pl.read_csv(DATA_DIR_ORIGINAL.joinpath('diagnoses_icd.csv'))
    icd_code_embedings = pl.read_csv(DATA_DIR_ROOT / 'icd-10-cm-2022-1000.csv').select(pl.all().exclude('desc'))

    # convert icd-9 to icd-10
    icd_code_mapper = Mapper()
    ICD_9_TO_10_MAP = partial(icd_code_mapper.map, source='icd9', target='icd10')

    diagnoses_data = diagnoses_data.with_columns(
        pl.when(pl.col('icd_version') == 9).then(pl.col('icd_code').map_elements(ICD_9_TO_10_MAP))
        .otherwise(pl.col('icd_code')).alias('icd_code'),
        # pl.col('icd_code').alias('original_code')
    ).drop('icd_version')

    # missing_codes = diagnoses_data.select(pl.col('original_code').filter(pl.col('icd_code').is_null()).unique().sort())
    missing_code_count = diagnoses_data.select(
        pl.col('icd_code').is_null().sum()
    )['icd_code'][0]
    print(f"Unable to convert {missing_code_count} codes to ICD-10. Dropping these samples.")
    diagnoses_data = diagnoses_data.filter(pl.col('icd_code').is_not_null())

    # For missing icd codes in the embedding aggregate all vectors of the next fitting icd code group
    missing_vector_icd_codes = (
        diagnoses_data
        .select(pl.col('icd_code')
                .filter(pl.col('icd_code')
                        .is_in(icd_code_embedings.select(pl.col('code').unique())
                               ).not_()).unique())
    )

    aggregated_missing_vector_icd_codes = []
    for missing_vector_icd_code in missing_vector_icd_codes['icd_code']:
        missing_embeddings = pl.DataFrame()
        chars_to_remove = 0
        while missing_embeddings.is_empty():
            chars_to_remove += 1
            missing_embeddings = icd_code_embedings.filter(
                pl.col('code').str.starts_with(missing_vector_icd_code[:-chars_to_remove])
            )

        aggregated_missing_vector_icd_codes.append(missing_embeddings.select(
            pl.lit(missing_vector_icd_code).alias('code'),
            pl.all().exclude('code').max(),
        ))

    aggregated_missing_vector_icd_codes = pl.concat(aggregated_missing_vector_icd_codes)
    icd_code_embedings = pl.concat([icd_code_embedings, aggregated_missing_vector_icd_codes])

    # Add embeddings to the icd codes
    diagnoses_data = diagnoses_data.join(
        icd_code_embedings,
        left_on='icd_code', right_on='code', how='left')

    diagnoses_data = diagnoses_data.group_by(['subject_id', 'hadm_id']).agg([
        pl.all().exclude('icd_code', 'seq_num').max(),
        pl.col('icd_code').str.concat(',').alias('icd_codes'),
    ])

    return diagnoses_data


def plot_admission_data(admission_data: pl.DataFrame) -> None:
    admission_plot_data = admission_data['nights_of_stay'].value_counts().sort(by='nights_of_stay')

    admission_plot_data_upper = admission_plot_data.select(
        (pl.col('nights_of_stay') * ((pl.col('nights_of_stay') >= 10) & (pl.col('nights_of_stay') < 100))).sum().alias(
            '10-99'),
        (pl.col('nights_of_stay') * (pl.col('nights_of_stay') >= 100)).sum().alias('100+'),
    )

    admission_plot_data_lower = admission_plot_data.filter(pl.col('nights_of_stay') < 10).select(
        pl.col('nights_of_stay').cast(pl.Int64).cast(pl.String),
        pl.col('count')
    ).transpose(column_names='nights_of_stay')

    admission_plot_data = pl.concat([admission_plot_data_lower, admission_plot_data_upper], how='horizontal')

    sns.barplot(admission_data, color='#5975a4')
    plt.xticks(np.arange(0, 12, 1), admission_plot_data.columns)
    plt.xlabel('Nights of Stays')
    plt.ylabel('Number of entries')
    plt.tight_layout()
    plt.show()


def get_length_of_stay(subject_and_hadm_ids: pl.DataFrame) -> pl.DataFrame:
    admission_data = pl.read_csv(DATA_DIR_ORIGINAL / 'admissions.csv').select(
        pl.col(['subject_id', 'hadm_id', 'admittime', 'dischtime'])
    ).join(subject_and_hadm_ids, on=['subject_id', 'hadm_id'], how='inner').with_columns(
        nights_of_stay=(pl.col('dischtime').str.to_date("%Y-%m-%d %H:%M:%S")
                        - pl.col('admittime').str.to_date("%Y-%m-%d %H:%M:%S")).cast(pl.Int64) / 86400000
    ).drop(['admittime', 'dischtime'])

    # print(admission_data['nights_of_stay'].describe())
    # plot_admission_data(admission_data)

    admission_data = admission_data.with_columns(
        nights_of_stay_group=pl
        .when(pl.col('nights_of_stay') < 3).then(0)
        .when(pl.col('nights_of_stay') < 7).then(1)
        .otherwise(2)
    )

    return admission_data


def read_mimic_extract_data() -> pl.DataFrame:
    pd.set_option("expand_frame_repr", False)
    hdf5_keys = ["codes", "interventions", "patients", "vitals_labs", "vitals_labs_mean"]
    for key in hdf5_keys:
        df: pd.DataFrame = pd.read_hdf(DATA_DIR_ROOT / 'all_hourly_data.h5', key=key)
        print(df.describe())


if __name__ == '__main__':
    omr_data_reshaped = reshape_omr_data()
    diagnoses_data_reshaped = reshape_diagnoses_data()
    length_of_stay = get_length_of_stay(diagnoses_data_reshaped.select(pl.col(['subject_id', 'hadm_id'])))

    all_data = diagnoses_data_reshaped.join(omr_data_reshaped, on='subject_id', how='left')
    all_data = all_data.join(length_of_stay, on=['subject_id', 'hadm_id'], how='inner')

    with pl.Config(tbl_cols=-1, tbl_width_chars=1000):
        print(all_data.describe())
        print(all_data.head(5))

    all_data.fill_null(-1).write_csv(DATA_DIR_ROOT / 'mimic_data_processed.csv')
    read_mimic_extract_data()
