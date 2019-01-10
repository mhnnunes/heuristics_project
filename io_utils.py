#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sys import argv
import seaborn as sns
# OS Imports
from os import getcwd
from os import listdir
from os.path import join
from os.path import isdir
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

sns.set()


def read_input(filename, header=True):
    if header:
        return pd.read_csv(filename, delimiter=',')
    else:
        return pd.read_csv(filename, delimiter=',', header=None)


def parse_breast_cancer(data):
    """ Pre-process data from the breast cancer wisconsin dataset
    https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29
    Arguments:
        data {pandas.DataFrame} -- DataFrame containing raw unprocessed data
    Returns:
        data {pandas.DataFrame} -- DataFrame containing data without
        labels and unnecessary info
    """
    data = data.drop('id', axis=1)
    data = data.drop('Unnamed: 32', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data.drop('diagnosis', axis=1).values, data['diagnosis']


def parse_iris(data):
    """ Pre-process data from the Iris dataset
    http://archive.ics.uci.edu/ml/datasets/Iris
    Arguments:
        data {pandas.DataFrame} -- DataFrame containing raw unprocessed data
    Returns:
        data {pandas.DataFrame} -- DataFrame containing data without
        labels and unnecessary info
    """
    data = data.drop('Id', axis=1)
    data['Species'] = data['Species'].map({'Iris-setosa': 0,
                                           'Iris-versicolor': 1,
                                           'Iris-virginica': 2})
    return data.drop('Species', axis=1).values, data['Species']


def parse_wine(data):
    """ Pre-process data from the Wine dataset
    http://archive.ics.uci.edu/ml/datasets/Wine
    Arguments:
        data {pandas.DataFrame} -- DataFrame containing raw unprocessed data
    Returns:
        data {pandas.DataFrame} -- DataFrame containing data without
        labels and unnecessary info
    """
    return data.drop(0, axis=1).values, data[0]


def parse_wine_quality(data):
    """ Pre-process data from the wine quality dataset
    https://archive.ics.uci.edu/ml/datasets/Wine+Quality
    Arguments:
        data {pandas.DataFrame} -- DataFrame containing raw unprocessed data
    Returns:
        data {pandas.DataFrame} -- DataFrame containing data without
        labels and unnecessary info
    """
    return data.drop('quality', axis=1).values, data['quality']


def parse_multi_challenge(data):
    return data.values


def generate_bivariate_gaussian(centers):
    for n_samples in [100, 500, 1000, 5000, 10000, 50000]:
        # Generate x blobs with 2 classes where the second blob contains
        # half positive samples and half negative samples. Probability in this
        # blob is therefore 0.5.
        centers = [(-2, 2), (0, 0), (2, 2)]
        X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                          centers=centers, shuffle=False, random_state=42)
        data = pd.concat([pd.DataFrame(X, columns=['X', 'Y']),
                          pd.DataFrame(y, columns=['class'])],
                         axis=1)
        data.to_csv('triangle_' + str(n_samples) + '.csv', index=False)
    for n_samples in [100, 500, 1000, 5000, 10000, 50000]:
        # Generate 3 blobs with 2 classes where the second blob contains
        # half positive samples and half negative samples. Probability in this
        # blob is therefore 0.5.
        centers = [(0, 0), (0, 5), (0, 10), (5, 0), (5, 5), (5, 10),
                   (10, 0), (10, 5), (10, 10)]
        X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                          centers=centers, shuffle=False, random_state=42)
        data = pd.concat([pd.DataFrame(X, columns=['X', 'Y']),
                          pd.DataFrame(y, columns=['class'])],
                         axis=1)
        data.to_csv('grid6_' + str(n_samples) + '.csv', index=False)


def parse_synthetic_dataset(data):
    try:
        labels = data['class']
    except Exception:
        labels = None
    return data[['X', 'Y']].values, labels


def plot_clustering_results(Y, results, heuristic, actual_clusters=None, fn=''):
    # PLOT CLUSTERING RESULT
    print('plotting results, ', fn)
    if actual_clusters is not None:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.scatter(Y[:, 0], Y[:, 1], c=results,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax1.set_title(heuristic)
        ax2.scatter(Y[:, 0], Y[:, 1], c=actual_clusters,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax2.set_title('Actual clusters')
        # plt.show()
        plt.savefig(fn)
    else:
        f, ax1 = plt.subplots(1, 1, sharey=True)
        ax1.scatter(Y[:, 0], Y[:, 1], c=results,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax1.set_title(heuristic)
        plt.savefig(fn)
    plt.close('all')


def parse_test_results(results_dir):
    all_results = pd.DataFrame(columns=['type', 'size',
                                        'method', 'k',
                                        'seed', 'time', 'ssq'])
    for f in listdir(results_dir):
        filename = join(results_dir, f)
        if not isdir(filename):
            print('reading file: ', filename)
            spl = f.split('.')[0].split('_')
            dataset_name = spl[0]
            # + '_' + spl[1]
            df = pd.read_csv(filename, delimiter=',')
            rows = df.shape[0]
            name = pd.DataFrame([dataset_name] * rows, columns=['type'])
            df = pd.concat([name, df], axis=1)
            all_results = pd.concat([all_results, df], axis=0,
                                    ignore_index=True)
    all_results.to_csv(join(results_dir, 'all_results.csv'), sep=',',
                       index=False)
    all_results['method'] = all_results['method'].map(lambda x: x.upper())
    # Make table for real results
    real_results = \
        all_results[(all_results['type'] != 'uniform') &
                    (all_results['type'] != 'grid6') &
                    (all_results['type'] != 'triangle')]

    synth_results = \
        all_results[(all_results['type'] == 'uniform') |
                    (all_results['type'] == 'grid6') |
                    (all_results['type'] == 'triangle')]
    # Write to file
    with open(join(results_dir, 'real_results_ssq.txt'), 'w') as f:
        c = ['mean', 'std', '50%']
        rr_df = real_results.groupby(['type', 'method'])[['time', 'ssq']]
        f.write(rr_df.describe()['ssq'][c].to_latex(float_format='%.2f'))
    # Time plot for real datasets
    g = sns.FacetGrid(real_results, col='method', palette='Dark2')
    g = (g.map(sns.pointplot, 'size', 'time', 'type',
               palette='Dark2').add_legend().set_axis_labels('Size',
                                                             'Time (s)'))
    plt.savefig(join(results_dir, 'plot_real_time.pdf'))
    # Plot MSSC by K for real datasets
    g = sns.FacetGrid(real_results, col='method', palette='Dark2')
    g = (g.map(sns.pointplot, 'k', 'ssq', 'type',
               palette='Dark2').add_legend().set_axis_labels('K', 'MSSC'))
    plt.savefig(join(results_dir, 'plot_k_real.pdf'))
    # Do the same for synthetic dataset
    with open(join(results_dir, 'synth_results_ssq.txt'), 'w') as f:
        c = ['mean', 'std', '50%']
        # Add grouping by size - UNCOMMENT below 
        # rr_df = synth_results.groupby(['size', 'type', 'method']).describe()
        # Remove grouping by size
        rr_df = synth_results.groupby(['type', 'method']).describe()
        f.write(rr_df[['ssq']]['ssq'][c].to_latex(float_format='%.2f'))
    # Time plot for synthetic datasets
    g = sns.FacetGrid(synth_results, col='method', palette='Dark2')
    g = (g.map(sns.pointplot, 'size', 'time', 'type',
               palette='Dark2').add_legend().set_axis_labels('Size',
                                                             'Time (s)'))
    plt.savefig(join(results_dir, 'plot_synth_time.pdf'))
    # Plot MSSC by K for synthetic datasets
    g = sns.FacetGrid(synth_results, col='method', row='type', palette='Dark2')
    g = (g.map(sns.pointplot, 'k', 'ssq', 'size', palette='Dark2')
         .add_legend(label_order=sorted(g._legend_data.keys(),
                     key=int)).set_axis_labels('K', 'MSSC'))
    plt.savefig(join(results_dir, 'plot_k_synth.pdf'))


if __name__ == "__main__":
    df = read_input(argv[1])
    print(df)
