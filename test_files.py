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


random_seed = 42


def test_datasets(datadir, outdir, verbose=False):
    dataset_dir = join(getcwd(), datadir)
    # result_dir = join(getcwd(), outdir)
    plots_dir = join(outdir, 'plots')

    for distribution in listdir(dataset_dir):
        # Test datasets generated using both distributions:
        # uniform and gaussian
        for filename in listdir(join(dataset_dir, distribution)):
            # Read each test file
            raw_data = read_input(join(dataset_dir, distribution, filename))
            X, labels = parse_synthetic_dataset(raw_data)
            if verbose:
                print("=========== Starting test on file: ", filename, '============')
            # Define values for k and put it in a loop
            results = pd.DataFrame(columns=['dataset', 'method', 'k',
                                            'time', 'ssq'])
            for k in range(1, 11):
                kmns = KMeans(X, random_seed, k, verbose)
                # BEGIN TEST: LLOYD HEURISTIC
                before = default_timer()
                try:
                    clusters_lloyd, ssq_lloyd = kmns.lloyd_heuristic()
                except Exception as e:
                    if verbose:
                        print(e)
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'lloyd', k,
                                                           np.nan, np.nan]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    continue
                else:
                    after = default_timer() - before
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'lloyd', k,
                                                           after, ssq_lloyd]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    plot_clustering_results(X, clusters_lloyd, 'lloyd', labels,
                                            plots_dir)
                # BEGIN TEST: MACQUEEN HEURISTIC
                before = default_timer()
                try:
                    clusters_mcq, ssq_mcq = kmns.macqueen_heuristic()
                except Exception as e:
                    if verbose:
                        print(e)
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'macqueen', k,
                                                           np.nan, np.nan]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    continue
                else:
                    after = default_timer() - before
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'macqueen', k,
                                                           after, ssq_mcq]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    plot_clustering_results(X, clusters_mcq, 'macqueen',
                                            labels, plots_dir)
                # BEGIN TEST: K-FURTHEST HEURISTIC
                before = default_timer()
                try:
                    clusters_kfu, ssq_kfu = kmns.k_furthest_initial_heuristic()
                except Exception as e:
                    if verbose:
                        print(e)
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'k-furthest', k,
                                                           np.nan, np.nan]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    continue
                else:
                    after = default_timer() - before
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'k-furthest', k,
                                                           after, ssq_kfu]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    plot_clustering_results(X, clusters_kfu, 'k-furthest',
                                            labels, plots_dir)
                # BEGIN TEST: K-POPULAR HEURISTIC
                before = default_timer()
                try:
                    clusters_kpp, ssq_kpp = kmns.k_popular_initial_heuristic()
                except Exception as e:
                    if verbose:
                        print(e)
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'k-popular', k,
                                                           np.nan, np.nan]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    continue
                else:
                    after = default_timer() - before
                    results = \
                        pd.concat([results, pd.DataFrame([[filename,
                                                           'k-popular', k,
                                                           after, ssq_kpp]],
                                                         columns=['dataset',
                                                                  'method',
                                                                  'k',
                                                                  'time',
                                                                  'ssq'])],
                                  axis=0)
                    plot_clustering_results(X, clusters_kpp, 'k-popular',
                                            labels, plots_dir)
                results.to_csv(join(outdir, 'results.csv'), index=False)


if __name__ == "__main__":
    parser = \
        argparse.ArgumentParser(description='Test script for KMeans.')
    parser.add_argument('-o', '--output', type=str,
                        default='testresults', help='output directory')
    parser.add_argument('-s', '--source', type=str,
                        default='dataset/real', help='input directory')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='increase output verbosity')
    args = parser.parse_args()

    test_datasets(args.source, args.output, args.verbose)
