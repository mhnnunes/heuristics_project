#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sys import argv
from io_utils import read_input
from io_utils import plot_clustering_results
from io_utils import parse_breast_cancer
from sklearn.manifold import TSNE  # Dimensionality reduction for visualization
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances


class KMeans(object):
    """ This class implements several heuristics for the
    Euclidean Minimum Sum-of-Squares Clustering (MSCC) problem.
    This problem is also commonly known as K-Means.
    """

    def __init__(self, data, seed, k, verbose=False):
        self.k = k
        # Data will be passed as a numpy 2D array
        self.data = data
        self.seed = seed
        self.verbose = verbose
        # Set random generator seed
        np.random.seed(self.seed)

    def calculate_distance_between_pairs(self):
        self.distances = euclidean_distances(self.data)

    def __calculate_sum_of_squares(self):
        # Calculate distance between points and clusters
        npoints = self.data.shape[0]
        points = np.arange(npoints)
        # Return sum of squares of distances between points and their
        # respective center (cluster)
        return np.sum(np.square(self.distances[points, self.clusters[points]]))

    def __reassign_points_to_clusters(self, centers_indexes, npoints):
        # Initially all points are in cluster 0
        clusters = np.zeros(npoints, dtype=int)
        for point in range(npoints):
            # Get distances from point to the centers, in the order:
            # [dist_to_center_0, dist_to_center_1, ...]
            distance_to_centers = self.distances[point, centers_indexes]
            # Get index of smallest distance
            closest_center = np.argmin(distance_to_centers)
            # Assign the point to the cluster with center on smallest distance
            clusters[point] = int(centers_indexes[closest_center])
        return clusters

    def lloyd_heuristic(self, threshold=10):
        # If the 'distances' variable does not exist, make it
        if not hasattr(self, 'distances'):
            self.calculate_distance_between_pairs()

        if self.verbose:
            print('Beginning of lloyd_heuristic')

        npoints = self.data.shape[0]
        # Initially all points are in cluster 0
        self.clusters = np.zeros(npoints, dtype=int)
        # Define k initial clusters randomly
        #    - Choose k points randomly
        # Choose k indexes from data
        centers_indexes = np.random.randint(npoints, size=self.k)
        # Use centers to index distance matrix, then sort
        self.clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                           npoints)

        # Number of points that changed cluster from one iteration to another
        changed_cluster_prev = 1
        changed_cluster_cur = 0
        # Mark the number of iterations in which the number of points that
        # have changed cluster is the same from the last iteration
        nochange = 0
        iteration = 1
        while nochange < threshold:
            # print("Iteration:: ", iteration)
            new_centers_indexes = []
            changed_cluster_cur = 0
            for center in centers_indexes:
                # Get indexes of points assigned to center
                points_in_cluster = np.where(self.clusters == center)[0]
                # Calculate centroid
                centroid = np.mean(self.data[points_in_cluster, :], axis=0)
                # print(centroid)
                # Get closest point to centroid
                closest_point_index = \
                    np.argmin(euclidean_distances(X=self.data,
                                                  Y=centroid.reshape(1, -1)))
                new_centers_indexes.append(closest_point_index)
            # end for loop
            centers_indexes = np.array(new_centers_indexes)
            # Reassign points to new clusters
            cur_clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                              npoints)
            # Count how many points changed clusters
            changed_cluster_cur = \
                np.count_nonzero(self.clusters - cur_clusters)
            if self.verbose:
                print(changed_cluster_cur, ' points changed cluster')
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
            if self.verbose:
                print('End of iteration: ', iteration)
            iteration += 1
            # print(changed_cluster_cur, ' points changed cluster')
        # Repeat until convergence (points stop changing clusters)
        # Calculate objective function value
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq

    def macqueen_heuristic(self, threshold=10):
        # If the 'distances' variable does not exist, make it
        if not hasattr(self, 'distances'):
            self.calculate_distance_between_pairs()

        if self.verbose:
            print('Beginning of macqueen_heuristic')

        npoints = self.data.shape[0]
        # Initially all points are in cluster 0
        self.clusters = np.zeros(npoints)
        # Define k initial clusters randomly
        #    - Choose k points randomly
        # Choose k indexes from data
        centers_indexes = np.random.randint(npoints, size=self.k)

        for center in centers_indexes:
            self.clusters[center] = center

        changed_cluster_prev = 1
        nochange = 0
        iteration = 1
        while nochange < threshold:
            # print("Iteration:: ", iteration)
            changed_cluster_cur = 0

            for point in range(npoints):
                # Get distances from point to the centers, in the order:
                # [dist_to_center_0, dist_to_center_1, ...]
                distance_to_centers = self.distances[point, centers_indexes]
                # Get index of smallest distance
                closest_center = np.argmin(distance_to_centers)
                # Assign the point to the cluster with center on smallest
                # distance
                if self.clusters[point] != centers_indexes[closest_center]:
                    changed_cluster_cur += 1
                self.clusters[point] = centers_indexes[closest_center]
                # Calculate new centroid for cluster in which the point was
                # added
                points_in_cluster = \
                    np.where(self.clusters ==
                             centers_indexes[closest_center])[0]
                centroid = np.mean(self.data[points_in_cluster, :], axis=0)
                closest_point_index = \
                    np.argmin(euclidean_distances(X=self.data,
                                                  Y=centroid.reshape(1, -1)))
                # Update points that were already in the cluster that changed
                for p in points_in_cluster:
                    self.clusters[p] = closest_point_index

                # Remove old centroid and add new one to list of centers
                centers_indexes = np.append(centers_indexes,
                                            closest_point_index)
                centers_indexes = np.delete(centers_indexes, closest_center)

            if self.verbose:
                print(changed_cluster_cur, ' points changed cluster')
            # Check if there were any changes of clusters
            if changed_cluster_prev == changed_cluster_cur:
                nochange += 1
            changed_cluster_prev = changed_cluster_cur
            # print(changed_cluster_cur, ' points changed cluster')
            if self.verbose:
                print('End of iteration: ', iteration)
            iteration += 1
        # Calculate objective function value
        self.clusters = self.clusters.astype(int)
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq

    def k_furthest_initial_heuristic(self, threshold=10):
        # If the 'distances' variable does not exist, make it
        if not hasattr(self, 'distances'):
            self.calculate_distance_between_pairs()

        if self.verbose:
            print('Beginning of k_furthest_initial_heuristic')

        npoints = self.data.shape[0]
        # Initially all points are in cluster 0
        self.clusters = np.zeros(npoints, dtype=int)
        # Choose 1 point randomly
        centers_indexes = np.zeros(self.k, dtype=int)
        centers_indexes[0] = np.random.randint(npoints, size=1)
        # Assign to the k-1 remainder centers the indexes of the k-1 points
        # that are further away from the current center
        # argsort return the indexes which would sort the array in ascending
        # order, then we take the last k-1 elements of this array with
        # [(self.k-1):] and return it backwards with [::-1]
        # print(centers_indexes[0])
        if self.k > 1:
            centers_indexes[1:] = \
                np.argsort(self.distances[int(centers_indexes[0]), :])[-(self.k -
                                                                         1):][::-1]
        # From then we apply lloyds algorithm
        # Use centers to index distance matrix, then sort
        self.clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                           npoints)

        # Number of points that changed cluster from one iteration to another
        changed_cluster_prev = 1
        changed_cluster_cur = 0
        # Mark the number of iterations in which the number of points that
        # have changed cluster is the same from the last iteration
        nochange = 0
        iteration = 1
        while nochange < threshold:
            # print("Iteration:: ", iteration)
            new_centers_indexes = []
            changed_cluster_cur = 0
            for center in centers_indexes:
                # Get indexes of points assigned to center
                points_in_cluster = np.where(self.clusters == center)[0]
                # Calculate centroid
                centroid = np.mean(self.data[points_in_cluster, :], axis=0)
                # print(centroid)
                # Get closest point to centroid
                closest_point_index = \
                    np.argmin(euclidean_distances(X=self.data,
                                                  Y=centroid.reshape(1, -1)))
                new_centers_indexes.append(closest_point_index)
            # end for loop
            centers_indexes = np.array(new_centers_indexes)
            # Reassign points to new clusters
            cur_clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                              npoints)
            # Count how many points changed clusters
            changed_cluster_cur = \
                np.count_nonzero(self.clusters - cur_clusters)
            if self.verbose:
                print(changed_cluster_cur, ' points changed cluster')
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
            if self.verbose:
                print('End of iteration: ', iteration)
            iteration += 1
            # print(changed_cluster_cur, ' points changed cluster')
        # Repeat until convergence (points stop changing clusters)
        # Calculate objective function value
        self.clusters = self.clusters.astype(int)
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq

    def k_popular_initial_heuristic(self, threshold=10):
        # If the 'distances' variable does not exist, make it
        if not hasattr(self, 'distances'):
            self.calculate_distance_between_pairs()

        if self.verbose:
            print('Beginning of k_popular_initial_heuristic')

        npoints = self.data.shape[0]
        # Initially all points are in cluster 0
        self.clusters = np.zeros(npoints)

        # Calculate mean of distances
        meandist = np.mean(self.distances)

        # Find the number of neighbors with distance > mean that each point has
        morethanmean = np.where(self.distances > meandist)
        numneigh = np.zeros(npoints, dtype=int)

        for point in range(npoints):
            numneigh[point] = np.count_nonzero(morethanmean[0] == point)

        # Choose the k points with more neighbors as the center indexes
        centers_indexes = np.zeros(self.k, dtype=int)
        centers_indexes[0:] = \
            np.argsort(numneigh)[-(self.k):][::-1]

        # From then we apply lloyds algorithm
        # Use centers to index distance matrix, then sort
        self.clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                           npoints)

        # Number of points that changed cluster from one iteration to another
        changed_cluster_prev = 1
        changed_cluster_cur = 0
        # Mark the number of iterations in which the number of points that
        # have changed cluster is the same from the last iteration
        nochange = 0
        iteration = 1
        while nochange < threshold:
            # print("Iteration:: ", iteration)
            new_centers_indexes = []
            changed_cluster_cur = 0
            for center in centers_indexes:
                # Get indexes of points assigned to center
                points_in_cluster = np.where(self.clusters == center)[0]
                # Calculate centroid
                centroid = np.mean(self.data[points_in_cluster, :], axis=0)
                # print(centroid)
                # Get closest point to centroid
                closest_point_index = \
                    np.argmin(euclidean_distances(X=self.data,
                                                  Y=centroid.reshape(1, -1)))
                new_centers_indexes.append(closest_point_index)
            # end for loop
            centers_indexes = np.array(new_centers_indexes)
            # Reassign points to new clusters
            cur_clusters = self.__reassign_points_to_clusters(centers_indexes,
                                                              npoints)
            # Count how many points changed clusters
            changed_cluster_cur = \
                np.count_nonzero(self.clusters - cur_clusters)
            if self.verbose:
                print(changed_cluster_cur, ' points changed cluster')
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
            if self.verbose:
                print('End of iteration: ', iteration)
            iteration += 1
            # print(changed_cluster_cur, ' points changed cluster')
        # Repeat until convergence (points stop changing clusters)
        # Calculate objective function value
        self.clusters = self.clusters.astype(int)
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq


if __name__ == "__main__":
    filename = argv[1]
    data = read_input(filename)
    tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
    le = LabelEncoder()
    # pre-process data
    # pre-processing breast cancer data
    X, label = parse_breast_cancer(data)
    X = X.values
    print(X)
    Y = tsne.fit_transform(scale(X))
    heu = KMeans(X, 1, 2, True)
    print("LLOYD HEURISTIC")
    c, ssq = heu.lloyd_heuristic()
    # print(le.fit_transform(c))
    plot_clustering_results(Y, le.fit_transform(c),
                            'LLOYD', data['diagnosis'])
    # plot_clustering_results(Y, le.fit_transform(c), 'LLOYD')
    print('Sum of squares:: ', ssq)
    print('MACQUEEN HEURISTIC')
    d, ssq = heu.macqueen_heuristic()
    # print(d)
    plot_clustering_results(Y, le.fit_transform(d),
                            'MACQUEEN', label)
    # plot_clustering_results(Y, le.fit_transform(d), 'MACQUEEN')
    print('Sum of squares:: ', ssq)
    print('K FURTHEST HEURISTIC')
    e, ssq = heu.k_furthest_initial_heuristic()
    # print(e)
    plot_clustering_results(Y, le.fit_transform(e), 'K-FURTHEST',
                            label)
    # plot_clustering_results(Y, le.fit_transform(d), 'K-FURTHEST')
    print('Sum of squares:: ', ssq)
    print('K POPULAR HEURISTIC')
    f, ssq = heu.k_popular_initial_heuristic()
    plot_clustering_results(Y, le.fit_transform(f), 'K-POPULAR',
                            label)
    # plot_clustering_results(Y, le.fit_transform(d), 'K-POPULAR')
    print('Sum of squares:: ', ssq)
    # print('Sum of squares:: ', ssq)


# import pandas as pd
# from io_utils import read_input 
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import scale

# filename = 'dataset/breast_cancer.csv' 
# data = read_input(filename)
# pre-process data
# pre-processing breast cancer data
# data = data.drop('id', axis=1)
# data = data.drop('Unnamed: 32', axis=1)
# data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
# data = data.drop('diagnosis', axis=1)
# print(data)
# print(data.values)

# datas = pd.DataFrame(scale(data))
# datas.columns = list(data.columns)

# X = datas.values
# tsne = TSNE(verbose=1, perplexity=40, n_iter=4000)
# Y = tsne.fit_transform(X)
# kmns = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1, algorithm='auto')
# kY = kmns.fit_predict(X)
