#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from sys import argv


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


if __name__ == "__main__":
    df = read_input(argv[1])
    print(df)
