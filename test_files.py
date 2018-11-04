#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import numpy as np
import pandas as pd
from os import getcwd
from os import listdir
from os.path import join
from heuristics import KMeans
from io_utils import read_input
from timeit import default_timer
from io_utils import parse_synthetic_dataset
from io_utils import plot_clustering_results


def run_heuristic_on_dataset(func, X, k, h_name, filename, plots_dir, labels,
                             seed, verbose):
    dataset_name = filename.split('.')[0]
    dataset_size = int(dataset_name.split('_')[1])
    # BEGIN TEST: LLOYD HEURISTIC
    before = default_timer()
    try:
        print('before calling', h_name)
        clusters, ssq = func()
        print('after calling', h_name)
    except Exception as e:
        print('EXCEPTION ON FILE', filename, e)
        if verbose:
            print(e)
        return False, pd.DataFrame([[dataset_size,
                                     h_name,
                                     k,
                                     np.nan,
                                     np.nan]],
                                   columns=['size',
                                            'method',
                                            'k',
                                            'seed',
                                            'time',
                                            'ssq'])
    else:
        after = default_timer() - before
        plot_clustering_results(X, clusters, h_name, labels,
                                join(plots_dir, dataset_name +
                                     '_k=' + str(k) + '_' + h_name + '.png'))
        return True, pd.DataFrame([[dataset_size,
                                    h_name,
                                    k,
                                    after,
                                    ssq]],
                                  columns=['size',
                                           'method',
                                           'k',
                                           'seed',
                                           'time',
                                           'ssq'])


def test_datasets(datadir, outdir, verbose=False):
    dataset_dir = join(getcwd(), datadir)
    # result_dir = join(getcwd(), outdir)
    plots_dir = join(getcwd(), outdir, 'plots')

    for distribution in listdir(dataset_dir):
        # Test datasets generated using both distributions:
        # uniform and gaussian
        for filename in sorted(listdir(join(dataset_dir, distribution))):
            # Read each test file
            if verbose:
                print("=========== Starting test on file: ",
                      filename, '============')
            print('before reading data')
            raw_data = read_input(join(dataset_dir, distribution, filename))
            X, labels = parse_synthetic_dataset(raw_data)
            print('after reading data')
            # Define values for k and put it in a loop
            results = pd.DataFrame(columns=['size', 'method', 'k', 'seed',
                                            'time', 'ssq'])
            for k in range(1, 11):
                for random_seed in range(30):
                    kmns = KMeans(X, random_seed, k, verbose)
                    heuristics_list = \
                        [(kmns.lloyd_initial_heuristic, 'lloyd'),
                         # (kmns.macqueen_heuristic, 'macqueen'),
                         (kmns.k_means_plus_plus, 'kmeans++'),
                         (kmns.k_furthest_initial_heuristic, 'k-furthest'),
                         (kmns.k_popular_initial_heuristic, 'k-popular')]
                    for heuristic, h_name in heuristics_list:
                        success, result = \
                            run_heuristic_on_dataset(heuristic, X, k,
                                                     h_name, filename,
                                                     plots_dir, labels,
                                                     random_seed, verbose)
                        if verbose:
                            print ('result shape: ', result.shape)
                        results = pd.concat([results, result], axis=0)
                        if verbose:
                            print ('total results shape: ', results.shape)
            # Removed polluting code
            results.to_csv(join(outdir, filename.split('.')[0] + '_' +
                                'results.csv'),
                           index=False)


if __name__ == "__main__":
    parser = \
        argparse.ArgumentParser(description='Test script for KMeans.')
    parser.add_argument('-o', '--output', type=str,
                        default='testresults', help='output directory')
    parser.add_argument('-s', '--source', type=str,
                        default='dataset/synthetic', help='input directory')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='increase output verbosity')
    args = parser.parse_args()

    test_datasets(args.source, args.output, args.verbose)
