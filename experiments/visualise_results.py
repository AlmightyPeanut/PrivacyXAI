import json
import os
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_rel

from experiments.utils import RESULTS_PATH

RESULTS_PATH = RESULTS_PATH / 'results_0804'
XAI_RESULTS_PATH = RESULTS_PATH / 'xai'
MIA_RESULTS_PATH = RESULTS_PATH / 'mia'

PLOT_ORDER = [
    'Centr.',
    'Centr. + DP 3',
    'Centr. + DP 1',
    'Centr. + DP .5',
    'FL 5',
    'FL 5 + DP 3',
    'FL 5 + DP 1',
    'FL 5 + DP .5',
    'FL 10',
    'FL 10 + DP 3',
    'FL 10 + DP 1',
    'FL 10 + DP .5',
    'FL 20',
    'FL 20 + DP 3',
    'FL 20 + DP 1',
    'FL 20 + DP .5',
    'FL 50',
    'FL 50 + DP 3',
    'FL 50 + DP 1',
    'FL 50 + DP .5',
]


def extract_model_name_from_file_name(file_name: str) -> str:
    model_name = re.sub(r'\.json', '', file_name)
    model_name = re.sub(r'\.csv', '', model_name)
    model_name = re.sub(r'_fold=\d+', '', model_name)
    model_name = re.sub(r'fl_model_metrics', 'fl', model_name)
    model_name = re.sub(r'centralised_model_metrics', 'central', model_name)
    model_name = re.sub(r'_fl=True', '', model_name)
    model_name = re.sub(r'_fl=False', '', model_name)
    model_name = re.sub(r'_rounds=\d+', '', model_name)
    model_name = re.sub(r'_privatised=False', '', model_name)
    model_name = re.sub(r'_privatised=True', '_dp', model_name)
    model_name = re.sub(r'_delta=[\d.]+', '', model_name)
    model_name = re.sub(r'_max_grad_norm=[\d.]+', '', model_name)
    model_name = re.sub(r'epsilon', 'eps', model_name)
    return model_name


def map_model_name_to_display_name(model_name: str) -> str:
    display_name = ''

    if 'fl_' in model_name:
        number_of_clients = int(re.findall(r'clients=\d+', model_name)[0].replace('clients=', ''))
        display_name += f'FL {number_of_clients}'
    else:
        display_name += 'Centr.'

    if 'privatised' in model_name:
        epsilon = float(re.findall(r'eps=[\d.]+', model_name)[0].replace('eps=', ''))
        if epsilon < 1:
            epsilon_str = str(epsilon)[1:]
        else:
            epsilon_str = str(epsilon)[:-2]
        display_name += f' + DP {epsilon_str}'

    return display_name


def plot_metric_results(ax: plt.Axes, data: pd.DataFrame, metric_name, remove_legend: bool = True):
    sns.boxplot(data,
                x='model_name',
                y=metric_name,
                hue='Model Type',
                order=PLOT_ORDER,
                ax=ax)

    ax.set(xlabel='')
    if remove_legend:
        ax.get_legend().remove()

    for i in range(1, len(PLOT_ORDER) // 4):
        ax.axvline(i * 4 - .5, linestyle='--', color='k')


def visualise_training_results():
    results = dict()
    for file in os.listdir(RESULTS_PATH):
        if not file.endswith('.json'):
            continue

        with open(RESULTS_PATH / file, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        model_name = extract_model_name_from_file_name(file)

        if model_name not in results:
            results[model_name] = []

        if file.startswith('fl_'):
            last_iteration_results = {
                'LR': data['LR'][-1][1],
                'NN': data['NN'][-1][1],
            }
            results[model_name].append(last_iteration_results)
        else:
            results[model_name].append(data)

    all_data_lr = []
    all_data_nn = []
    for model_name, result in results.items():
        lr_results = []
        nn_results = []
        for fold_data in result:
            lr_results.append(fold_data['LR'])
            nn_results.append(fold_data['NN'])
        dataframe_lr = pd.DataFrame.from_dict(lr_results).reset_index(names='fold')
        dataframe_lr['model_name'] = model_name
        dataframe_lr['Model Type'] = 'LR'
        all_data_lr.append(dataframe_lr)

        dataframe_nn = pd.DataFrame.from_dict(nn_results).reset_index(names='fold')
        dataframe_nn['model_name'] = model_name
        dataframe_nn['Model Type'] = 'NN'
        all_data_nn.append(dataframe_nn)

    avg_all_data_lr = pd.concat(all_data_lr, ignore_index=True)[['AUROC', 'Acc', 'Binary F1 Score']]#.mean()
    # print(avg_all_data_lr)
    avg_all_data_nn = pd.concat(all_data_nn, ignore_index=True)[['AUROC', 'Acc', 'Binary F1 Score']]#.mean()
    # print(avg_all_data_nn)
    print(avg_all_data_lr.describe())
    print(avg_all_data_nn.describe())
    return

    all_data = all_data_lr + all_data_nn
    plot_training_results(all_data)


def calculate_statistical_significance(data: pd.DataFrame, compare_identifier: str) -> pd.DataFrame:
    results = {}
    names = data[compare_identifier].unique()
    for name_1 in names:
        if name_1 not in results:
            results[name_1] = {}
        for name_2 in names:
            if name_2 in results:
                if name_1 in results[name_2]:
                    continue
            else:
                results[name_2] = {}

            if name_1 == name_2:
                results[name_1][name_2] = 1.
                continue

            significance = ttest_rel(
                data[data[compare_identifier] == name_1].drop(compare_identifier, axis=1).to_numpy(),
                data[data[compare_identifier] == name_2].drop(compare_identifier, axis=1).to_numpy())
            pvalue = significance.pvalue[0]
            if np.isnan(pvalue):
                pvalue = 1.
            results[name_1][name_2] = pvalue
            results[name_2][name_1] = pvalue
    return pd.DataFrame.from_dict(results)


def annotate_ax_with_group_names_x(ax: plt.Axes, group_size: int, group_names: list[str], group_name_margin: float):
    x_ticks = ax.get_xticks()

    new_x_tick_labels = [re.sub(r"(Centr.|FL \d+)( \+ DP )?", '', x_label.get_text())
                         for x_label in ax.get_xticklabels()]
    new_x_tick_labels = [label if label != '' else 'n/a' for label in new_x_tick_labels]
    ax.set_xticklabels(new_x_tick_labels)
    plt.xticks(rotation=0, ha='center')

    arrow_properties = dict(
        connectionstyle='angle, angleA=180, angleB=90, rad=0',
        arrowstyle='-',
        shrinkA=3,
        shrinkB=20,
        lw=1,
    )

    for group_id in range(len(x_ticks) // group_size):
        group_name = group_names[group_id]

        y_lim_low = ax.get_ylim()[0]
        first_x_tick = x_ticks[group_id * 4]
        last_x_tick = x_ticks[group_id * 4 + 3]
        text_xy = ((first_x_tick + last_x_tick) / 2, y_lim_low + group_name_margin)

        ax.annotate(group_name, xy=(first_x_tick, y_lim_low), xytext=text_xy, horizontalalignment='center',
                    arrowprops=arrow_properties)
        ax.annotate(group_name, xy=(last_x_tick, y_lim_low), xytext=text_xy, horizontalalignment='center',
                    arrowprops=arrow_properties)


def annotate_ax_with_group_names_y(ax: plt.Axes, group_size: int, group_names: list[str]):
    y_ticks = ax.get_yticks()

    new_y_tick_labels = [re.sub(r"(Centr.|FL \d+)( \+ DP )?", '', y_label.get_text())
                         for y_label in ax.get_yticklabels()]
    new_y_tick_labels = [label if label != '' else 'n/a' for label in new_y_tick_labels]
    ax.set_yticklabels(new_y_tick_labels)

    arrow_properties = dict(
        connectionstyle='angle, angleA=90, angleB=180, rad=0',
        arrowstyle='-',
        shrinkA=1,
        shrinkB=30,
        lw=1,
    )

    for group_id in range(len(y_ticks) // group_size):
        group_name = group_names[group_id]

        x_lim_low = ax.get_xlim()[0]
        first_y_tick = y_ticks[group_id * 4]
        last_y_tick = y_ticks[group_id * 4 + 3]
        text_xy = (x_lim_low - 1.75, (first_y_tick + last_y_tick) / 2)

        ax.annotate(group_name, xy=(x_lim_low, first_y_tick), xytext=text_xy, horizontalalignment='center',
                    rotation=90, rotation_mode='anchor',
                    arrowprops=arrow_properties)
        ax.annotate(group_name, xy=(x_lim_low, last_y_tick), xytext=text_xy, horizontalalignment='center',
                    rotation=90, rotation_mode='anchor',
                    arrowprops=arrow_properties)


# TODO: Farben anpassen
STAT_SIGNIFICANCE_COLOR_MAP = sns.color_palette('blend:#008585,#008585', as_cmap=True)
STAT_SIGNIFICANCE_COLOR_MAP.set_over('#c7522a')


def visualise_statistical_significance(data: pd.DataFrame, metric_name: str, plot_order: list[str],
                                       group_size: int, file_appendix: str = '') -> None:
    fig, ax = plt.subplots(1, 1)
    sns.heatmap(data[plot_order].loc[plot_order], vmin=.0, center=0.04, vmax=0.05,
                cmap=STAT_SIGNIFICANCE_COLOR_MAP, cbar=False, ax=ax)

    for i in range(1, len(plot_order) // group_size):
        ax.axvline(i * group_size, linestyle='--', color='k')
        ax.axhline(i * group_size, linestyle='--', color='k')

    # add groups to labels
    group_names = plot_order[::group_size]
    annotate_ax_with_group_names_x(ax, group_size, group_names, 2.)
    annotate_ax_with_group_names_y(ax, group_size, group_names)
    # ax.text(-1., -.5, 'ε', fontweight='bold')
    ax.text(-.75, 21.2, 'ε')

    plt.tight_layout()
    # plt.show()
    fig.savefig(RESULTS_PATH / f'{metric_name}_statistical_significance{file_appendix}.png')
    plt.close(fig)


def plot_training_results(data):
    all_data_frame = pd.concat(data, ignore_index=True).drop('fold', axis=1)
    all_data_frame['model_name'] = all_data_frame['model_name'].map(map_model_name_to_display_name)
    # all_data_frame = all_data_frame.melt(id_vars=['model_name', 'model_type'], var_name='Performance Metric')

    metrics = ['AUROC', 'Acc', 'Binary F1 Score']  # Excluded: 'Macro F1 Score'
    plt.rcParams.update({'font.size': 10})

    # visualise results
    fig, axes = plt.subplots(nrows=len(metrics), sharex=True, sharey=True, figsize=(7, 6))
    for ax_id, metric in enumerate(metrics):
        plot_data = all_data_frame[[metric, 'model_name', 'Model Type']]
        if metric == 'Binary F1 Score':
            plot_data = plot_data.rename(columns={
                'Binary F1 Score': 'F1 Score'
            })
            metric = 'F1 Score'

        if ax_id != 0:
            plot_metric_results(axes[ax_id], plot_data, metric)
        else:
            plot_metric_results(axes[ax_id], plot_data, metric, False)

    group_names = PLOT_ORDER[::4]
    annotate_ax_with_group_names_x(axes[-1], 4, group_names, -0.06)

    plt.tight_layout()
    # plt.show()
    plt.savefig(RESULTS_PATH / 'training_results.png')
    plt.clf()

    # calculate + visualise stat significance
    plt.rcParams.update({'font.size': 12})
    for metric in metrics:
        statistical_significance_lr = calculate_statistical_significance(
            all_data_frame[all_data_frame['Model Type'] == 'LR'][[metric, 'model_name']],
            'model_name'
        )
        visualise_statistical_significance(statistical_significance_lr, metric, PLOT_ORDER, 4, '_LR')

        statistical_significance_nn = calculate_statistical_significance(
            all_data_frame[all_data_frame['Model Type'] == 'NN'][[metric, 'model_name']],
            'model_name'
        )
        visualise_statistical_significance(statistical_significance_nn, metric, PLOT_ORDER, 4, '_NN')


def visualise_xai_results():
    all_data_lr = []
    all_data_nn = []
    for file in os.listdir(XAI_RESULTS_PATH):
        if not file.endswith('.csv'):
            continue

        data = pd.read_csv(XAI_RESULTS_PATH / file)
        if data.empty:
            continue
        data.rename(columns={
            'Unnamed: 0': "XAI Metric"
        }, inplace=True)

        data = data.melt(id_vars=data.columns[0], value_vars=data.columns[1:],
                         var_name='XAI Method', value_name='Value')

        model_name = extract_model_name_from_file_name(file)
        data['model_name'] = model_name

        if model_name.startswith('nn'):
            model_type = 'NN'
        else:
            model_type = 'LR'
        data['Model Type'] = model_type

        if model_name.startswith('nn'):
            all_data_nn.append(data)
        else:
            all_data_lr.append(data)

    all_data = pd.concat(all_data_lr + all_data_nn, ignore_index=True)
    plot_xai_results(all_data, '_all')

    # all_data_lr = pd.concat(all_data_lr, ignore_index=True)
    # all_data_lr["Model Type"] = all_data_lr['XAI Method'].replace({
    #     'IntegratedGradients': 'IG',
    #     'FeaturePermutation': 'FP',
    # })
    # plot_xai_results(all_data_lr, '_lr')
    #
    # all_data_nn = pd.concat(all_data_nn, ignore_index=True)
    # all_data_nn["Model Type"] = all_data_nn['XAI Method'].replace({
    #     'IntegratedGradients': 'IG',
    #     'FeaturePermutation': 'FP',
    # })
    # plot_xai_results(all_data_nn, '_nn')


def plot_xai_results(data: pd.DataFrame, file_appendix: str = ''):
    data['model_name'] = data['model_name'].map(map_model_name_to_display_name)

    # TODO: add all metrics
    XAI_METRIC_GROUPS = {
        'Faithfulness': ['Faithfulness Estimate', 'Monotonicity', 'Sufficiency'],
        'Robustness': ['Avg Sensitivity', 'Local Lipschitz Estimate', 'Max Sensitivity'],
        'Complexity': ['Complexity', 'Effective Complexity', 'Sparseness'],
        'Sensitivity': ['Efficient MPRT', 'MPRT', 'Smooth MPRT'],
        'Axiomatic': ['Completeness', 'Non-Sensitivity'],
    }

    plt.rcParams.update({'font.size': 12})
    for metric_group_name, metric_group in XAI_METRIC_GROUPS.items():
        # fig = plt.figure(figsize=(7, 2*len(metric_group)))
        # axes = fig.add_axes((0.11, 0.3, .85, .65))
        fig, axes = plt.subplots(nrows=len(metric_group), sharex=True, figsize=(7, 2 * len(metric_group)))
        if isinstance(axes, plt.Axes):
            axes = np.array([axes])
        for ax_id, metric in enumerate(metric_group):
            plot_data = data.loc[data['XAI Metric'] == metric].loc[data['XAI Method'] == "Lime"]

            if ax_id != 0:
                plot_metric_results(axes[ax_id], plot_data, 'Value')
            else:
                plot_metric_results(axes[ax_id], plot_data, 'Value', False)

            # statistical_significance_lr = calculate_statistical_significance(
            #     plot_data[plot_data['Model Type'] == 'LR'][['Value', 'model_name']],
            #     'model_name'
            # )
            # visualise_statistical_significance(statistical_significance_lr, metric, PLOT_ORDER, 4, '_LR_XAI_FP')
            #
            # statistical_significance_nn = calculate_statistical_significance(
            #     plot_data[plot_data['Model Type'] == 'NN'][['Value', 'model_name']],
            #     'model_name'
            # )
            # visualise_statistical_significance(statistical_significance_nn, metric, PLOT_ORDER, 4, '_NN_XAI_FP')

            axes[ax_id].set_ylabel(metric)

        margin = -.025
        if metric_group_name == 'Faithfulness':
            # axes[0].set_ylim(top=0.05)
            # axes[0].set_ylabel("Faithfulness\nEstimate")
            margin = -.11
        elif metric_group_name == 'Robustness':
            # axes[1].set_ylabel("Local Lipschitz\nEstimate")
            # axes[0].set_ylim(top=2.)
            # axes[2].set_ylim(top=4.5)
            margin = -20.
            # fig.subplots_adjust(bottom=.0, top=1., hspace=.0, wspace=.0)
        elif metric_group_name == 'Complexity':
            margin = -.0
            # axes[2].set_ylim(bottom=.99)
        elif metric_group_name == 'Sensitivity':
            margin = -.1
        elif metric_group_name == 'Axiomatic':
            margin = -.15
        annotate_ax_with_group_names_x(axes[-1], 4, PLOT_ORDER[::4], margin)

        plt.tight_layout()
        # plt.show()
        plt.savefig(RESULTS_PATH / f'xai_results_{metric_group_name}{file_appendix}_FP.png')
        plt.clf()


def plot_mia_results(data: pd.DataFrame):
    data['model_name'] = data['model_name'].map(map_model_name_to_display_name)

    metrics = data.columns.drop(['model_name', 'Model Type']).unique()

    fig, axes = plt.subplots(len(metrics), 1, sharex=True, figsize=(7, 6))
    plt.rcParams.update({'font.size': 10})
    for ax_id, metric in enumerate(metrics):
        plot_data = data[[metric, 'model_name', 'Model Type']]
        if ax_id == 0:
            plot_metric_results(axes[ax_id], plot_data, metric, False)
        else:
            plot_metric_results(axes[ax_id], plot_data, metric)

    group_names = PLOT_ORDER[::4]
    annotate_ax_with_group_names_x(axes[-1], 4, group_names, -0.25)

    plt.tight_layout()
    # plt.show()
    plt.savefig(RESULTS_PATH / 'mia_results.png')
    plt.clf()

    # calculate + visualise stat significance
    plt.rcParams.update({'font.size': 12})
    for metric in metrics:
        statistical_significance_lr = calculate_statistical_significance(
            data[data['Model Type'] == 'LR'][[metric, 'model_name']],
            'model_name'
        )
        visualise_statistical_significance(statistical_significance_lr, metric, PLOT_ORDER, 4, '_LR_mia')

        statistical_significance_nn = calculate_statistical_significance(
            data[data['Model Type'] == 'NN'][[metric, 'model_name']],
            'model_name'
        )
        visualise_statistical_significance(statistical_significance_nn, metric, PLOT_ORDER, 4, '_NN_mia')


def visualise_mia_results():
    all_data_lr = []
    all_data_nn = []
    for file in os.listdir(MIA_RESULTS_PATH):
        if not file.endswith('.json'):
            continue

        with open(MIA_RESULTS_PATH / file, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        model_name = extract_model_name_from_file_name(file)

        new_data = {
            'model_name': model_name,
            'Accuracy': data['attack_accuracy'],
            'Precision': data['attack_precision'],
            'Recall': data['attack_recall'],
        }

        if model_name.startswith('nn'):
            new_data['Model Type'] = 'NN'
            all_data_nn.append(new_data)
        else:
            new_data['Model Type'] = 'LR'
            all_data_lr.append(new_data)

    all_data = pd.DataFrame(all_data_lr + all_data_nn)
    plot_mia_results(all_data)


if __name__ == '__main__':
    visualise_training_results()
    visualise_xai_results()
    visualise_mia_results()
