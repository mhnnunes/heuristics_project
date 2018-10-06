#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
from sys import argv
import matplotlib.pyplot as plt
from sklearn.dataset import make_blobs


def read_input(filename):
    return pd.read_csv(filename, delimiter=',')


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
    return data.drop('diagnosis', axis=1), data['diagnosis']


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
        data.to_csv('grid6_' + str(n_samples) + '.csv', index=False)


def parse_synthetic_dataset(data):
    return data[['X', 'Y']].values, data['class']


def plot_clustering_results(Y, results, heuristic, actual_clusters=None):
    # PLOT CLUSTERING RESULT
    if actual_clusters is not None:
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.scatter(Y[:, 0], Y[:, 1], c=results,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax1.set_title(heuristic)
        ax2.scatter(Y[:, 0], Y[:, 1], c=actual_clusters,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax2.set_title('Actual clusters')
        # plt.show()
        plt.savefig(str(heuristic) + '.png')
    else:
        f, ax1 = plt.subplots(1, 1, sharey=True)
        ax1.scatter(Y[:, 0], Y[:, 1], c=results,
                    cmap="jet", edgecolor="None", alpha=0.35)
        ax1.set_title(heuristic)
        plt.savefig(str(heuristic) + '.png')


if __name__ == "__main__":
    df = read_input(argv[1])
    print(df)
