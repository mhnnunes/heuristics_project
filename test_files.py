#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os import getcwd
from os import listdir
from os.path import join
from datetime import datetime
from heuristics import KMeans
from io_utils import read_input
from timeit import default_timer
from io_utils import parse_synthetic_dataset
from io_utils import plot_clustering_results


random_seed = 42


def test_synthetic_datasets():
    dataset_dir = join(getcwd(), 'dataset', 'synthetic')
    result_dir = join(getcwd(), 'testresults')
    plots_dir = join(result_dir, 'plots')

    for distribution in listdir(dataset_dir):
        # Test datasets generated using both distributions:
        # uniform and gaussian
        for filename in listdir(join(dataset_dir, distribution)):
            # Read each test file
            raw_data = read_input(filename)
            X, labels = parse_synthetic_dataset(raw_data)
            # Define values for k and put it in a loop
            for k in range(1, 11):
                kmns = KMeans(X, random_seed, k)
                before = default_timer()
                clusters, ssq = kmns.lloyd_heuristic()
                after = default_timer() - before
                before = default_timer()
                clusters, ssq = kmns.macqueen_heuristic()
                after = default_timer() - before
                before = default_timer()
                clusters, ssq = kmns.k_furthest_initial_heuristic()
                after = default_timer() - before
                before = default_timer()
                clusters, ssq = kmns.k_popular_initial_heuristic()
                after = default_timer() - before
