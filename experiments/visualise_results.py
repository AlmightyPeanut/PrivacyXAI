import json
import os
import re

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from experiments.utils import RESULTS_PATH

# RESULTS_PATH = RESULTS_PATH / 'training_results_0627'


def extract_model_name_from_file_name(file_name: str) -> str:
    model_name = re.sub(r'\.json', '', file_name)
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

    if model_name.startswith('fl_'):
        number_of_clients = int(re.findall(r'clients=\d+', model_name)[0].replace('clients=', ''))
        display_name += f'FedAvg ({number_of_clients} clients)'
    else:
        display_name += 'Centralised'

    if 'privatised' in model_name:
        epsilon = float(re.findall(r'eps=[\d.]+', model_name)[0].replace('eps=', ''))
        display_name += f'\n + DP (ε={epsilon})'

    return display_name


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
        all_data_lr.append(dataframe_lr)

        dataframe_nn = pd.DataFrame.from_dict(nn_results).reset_index(names='fold')
        dataframe_nn['model_name'] = model_name
        all_data_nn.append(dataframe_nn)

    plot_training_results(all_data_lr, 'LR')
    plot_training_results(all_data_nn, 'NN')


def plot_training_results(data, model_type):
    all_data_frame = pd.concat(data, ignore_index=True).drop('fold', axis=1)
    all_data_frame['model_name'] = all_data_frame['model_name'].map(map_model_name_to_display_name)
    for ax in all_data_frame.boxplot(by='model_name', rot=90, layout=(1, 4), figsize=(12, 4), vert=False):
        plt.setp(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        ax.set_xlim([0, .8])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.figure.suptitle(f'Initial results ({model_type})')
    plt.tight_layout()
    plt.show()
    # plt.savefig(RESULTS_PATH / f'{model_type}_results.png')


def visualise_xai_results():
    all_data_lr = []
    all_data_nn = []
    for file in os.listdir(RESULTS_PATH / 'xai'):
        if not file.endswith('.json'):
            continue

        with open(RESULTS_PATH / 'xai' / file, 'r') as f:
            data = json.load(f)

        if not data:
            continue

        model_name = extract_model_name_from_file_name(file)

        new_data = []
        for xai_method_name, result in data.items():
            result["xai_method_name"] = xai_method_name
            result["model_name"] = model_name
            new_data.append(result)
        data = new_data

        if model_name.startswith('nn'):
            all_data_nn.extend(data)
        else:
            all_data_lr.extend(data)

    plot_xai_results(all_data_lr, 'LR')
    plot_xai_results(all_data_nn, 'NN')


def plot_xai_results(data, model_type):
    data = pd.DataFrame(data)
    data['model_name'] = data['model_name'].map(map_model_name_to_display_name)

    data_1 = data[data.columns[:-2]].stack().reset_index()
    data_1.columns = ['index', 'XAI Metric', 'Value']
    data = data[['xai_method_name', 'model_name']]
    t = pd.merge(data_1, data, left_on='index', right_index=True)

    g = sns.FacetGrid(data=t.sort_values(by='model_name', ascending=False), col='XAI Metric', row='xai_method_name',
                      sharex=False, col_order=t['XAI Metric'].sort_values(ascending=True).unique(),
                      row_order=t['xai_method_name'].sort_values(ascending=False).unique())
    g.map(plt.scatter, 'Value', 'model_name')
    g.set_xlabels('')
    g.set_ylabels('')
    g.set_titles('{col_name}\n{row_name}')
    plt.tight_layout()
    # plt.show()
    plt.savefig(RESULTS_PATH / f'{model_type}_xai.png')


if __name__ == '__main__':
    visualise_training_results()
    visualise_xai_results()
