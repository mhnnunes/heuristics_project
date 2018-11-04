#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sys import argv
from io_utils import read_input
from sklearn.manifold import TSNE  # Dimensionality reduction for visualization
from sklearn.preprocessing import scale
from io_utils import parse_breast_cancer
from io_utils import plot_clustering_results
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
        self.npoints = self.data.shape[0]
        # Set random generator seed
        np.random.seed(self.seed)

    def calculate_distance_between_pairs(self):
        self.distances = euclidean_distances(self.data)

    def __calculate_sum_of_squares(self):
        # Calculate distance between points and clusters
        points = np.arange(self.npoints)
        # Return sum of squares of distances between points and their
        # respective center (cluster)
        return np.sum(np.square(self.distances[points, self.clusters[points]]))

    def __reassign_points_to_clusters(self, centers_indexes):
        # Initially all points are in cluster 0
        clusters = np.zeros(self.npoints, dtype=int)
        for point in range(self.npoints):
            # Get distances from point to the centers, in order:
            # [dist_to_center_0, dist_to_center_1, ...]
            distance_to_centers = self.distances[point, centers_indexes]
            # Get index of smallest distance
            closest_center = np.argmin(distance_to_centers)
            # Assign the point to the cluster with center on smallest distance
            clusters[point] = int(centers_indexes[closest_center])
        return clusters

    def lloyd_local_search(self, centers_indexes, threshold):
        self.clusters = self.__reassign_points_to_clusters(centers_indexes)

        # Number of points that changed cluster from one iteration to another
        changed_cluster_prev = 1
        changed_cluster_cur = 0
        # Mark the number of iterations in which the number of points that
        # have changed cluster is the same from the last iteration
        nochange = 0
        iteration = 1
        while nochange < threshold:
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
            cur_clusters = self.__reassign_points_to_clusters(centers_indexes)
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
        # Repeat until convergence (points stop changing clusters)
        # Calculate objective function value
        self.clusters = self.clusters.astype(int)
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq

    def __init_clusters(self, heuristic_name=""):
        # Recalculating distance matrix every time so every
        # heuristic is evaluated in the same manner
        self.calculate_distance_between_pairs()

        if self.verbose:
            print('Beginning of ', heuristic_name)

        # Initially all points are in cluster 0
        self.clusters = np.zeros(self.npoints, dtype=int)

    def macqueen_heuristic(self, threshold=10):
        self.__init_clusters('macqueen_heuristic')

        # Define k initial clusters randomly
        #    - Choose k points randomly
        # Choose k indexes from data
        centers_indexes = np.random.randint(self.npoints, size=self.k)

        for center in centers_indexes:
            self.clusters[center] = center

        changed_cluster_prev = 1
        nochange = 0
        iteration = 1
        while nochange < threshold:
            # print("Iteration:: ", iteration)
            changed_cluster_cur = 0

            for point in range(self.npoints):
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

    def lloyd_initial_heuristic(self):
        self.__init_clusters('lloyd_heuristic')
        # Define k initial clusters randomly
        #    - Choose k points randomly
        # Choose k indexes from data
        centers_indexes = np.random.randint(self.npoints, size=self.k)
        # return centers_indexes
        self.clusters = self.__reassign_points_to_clusters(centers_indexes)
        return self.clusters, self.__calculate_sum_of_squares()

    def k_means_plus_plus(self):
        self.__init_clusters('k_means++ heuristic')
        points_indexes = np.array(list(range(self.npoints)))
        # Choose 1 point randomly
        centers_indexes = np.zeros(self.k, dtype=int)
        centers_indexes[0] = np.random.randint(self.npoints, size=1)

        if self.k > 1:
            for i in range(1, self.k):
                # Calculate D^2 for every data point
                d2 = self.distances[:, centers_indexes[:i]].min(axis=1)**2
                # Normalize D^2
                d2 = d2 / np.sum(d2)
                possible_points = np.setdiff1d(points_indexes,
                                               np.array(centers_indexes[:i]))
                centers_indexes[i] = np.random.choice(possible_points,
                                                      size=1,
                                                      p=(d2[possible_points]))
        if self.verbose:
            print("[k-means++] chosen centers: ", centers_indexes)

        # return centers_indexes
        self.clusters = self.__reassign_points_to_clusters(centers_indexes)
        return self.clusters, self.__calculate_sum_of_squares()

    def k_furthest_initial_heuristic(self):
        self.__init_clusters('k_furthest_initial_heuristic')
        # Choose 1 point randomly
        centers_indexes = np.zeros(self.k, dtype=int)
        centers_indexes[0] = np.random.randint(self.npoints, size=1)
        # Assign to the k-1 remainder centers the indexes of the k-1 points
        # that are further away from the current center
        # argsort return the indexes which would sort the array in ascending
        # order, then we take the last k-1 elements of this array with
        # [(self.k-1):] and return it backwards with [::-1]
        if self.k > 1:
            centers_indexes[1:] = \
                np.argsort(self.distances[int(centers_indexes[0]), :])[-(self.k -
                                                                         1):][::-1]
        # return centers_indexes
        self.clusters = self.__reassign_points_to_clusters(centers_indexes)
        return self.clusters, self.__calculate_sum_of_squares()

    def k_popular_initial_heuristic(self):
        self.__init_clusters('k_popular_initial_heuristic')

        # Calculate mean of distances
        meandist = np.mean(self.distances)

        # Find the number of neighbors with distance > mean that each point has
        morethanmean = np.where(self.distances > meandist)
        numneigh = np.zeros(self.npoints, dtype=int)

        for point in range(self.npoints):
            numneigh[point] = np.count_nonzero(morethanmean[0] == point)

        # Choose the k points with more neighbors as the center indexes
        centers_indexes = np.zeros(self.k, dtype=int)
        centers_indexes[0:] = \
            np.argsort(numneigh)[-(self.k):][::-1]

        # return centers_indexes
        self.clusters = self.__reassign_points_to_clusters(centers_indexes)
        return self.clusters, self.__calculate_sum_of_squares()

    def __tabu_neighborhood_search(self, initial_solution, tabu_list):
        print("initial_solution", initial_solution)
        print("initial_solution shape", initial_solution.shape[0])
        new_solution = np.zeros(initial_solution.shape[0], dtype=int)
        for index, center in enumerate(list(initial_solution)):
            # Evaluate deltaJ for each muk (center)
            points_in_cluster = np.where(self.clusters == center)[0]
            # print('points_in_cluster', points_in_cluster)
            # print('points_in_cluster shape', points_in_cluster.shape)
            delta_muk = self.distances[:, center]
            # print('delta_muk', delta_muk)
            deltaJ = np.zeros(points_in_cluster.shape[0])
            for index2, point in enumerate(list(points_in_cluster)):
                deltaJ[index2] = \
                    np.sum(-2 *
                           self.distances[points_in_cluster, center] *
                           delta_muk[point] + delta_muk[point] ** 2)
            assigned = False
            for point in points_in_cluster[np.argsort(deltaJ)]:
                if point not in tabu_list:
                    assigned = True
                    new_solution[index] = point
                    break
            if not assigned:
                # delete last entry on row k of tabu list:
                # same thing as using it again
                if self.verbose:
                    print('all points in cluster', index, 'are on tabu_list')
                new_solution[index] = tabu_list[self.k - 1, -1]
        return new_solution

    def tabu_search_metaheuristic(self, threshold=0.1, max_iter=10):
        self.__init_clusters('tabu_search_metaheuristic')

        best_solution = self.lloyd_initial_heuristic()
        self.clusters = self.__reassign_points_to_clusters(best_solution)

        # Mark the number of iterations taken to converge
        nochange = 0
        iteration = 1
        tabu_list = np.empty((self.k, 0))

        # Continue until the change in the objective function value is not
        # significant enough, for max_iter iterations
        while nochange < max_iter:  # TODO: Add here total max iterations num
            prev_ssq = self.__calculate_sum_of_squares()

            if self.verbose:
                print("initial sum of squares: ", prev_ssq)
            # Build new solution from neighborhood function
            new_solution = self.__tabu_neighborhood_search(best_solution,
                                                           tabu_list)
            # Add new solution to tabu_list
            tabu_list = np.hstack([tabu_list, new_solution.reshape(self.k, 1)])
            if self.verbose:
                print('new solution found on __tabu_neighborhood_search',
                      new_solution)
            # Reassign points to new clusters
            new_clusters = self.__reassign_points_to_clusters(new_solution)
            # Calculate new objective function value
            ssq = self.__calculate_sum_of_squares()
            if self.verbose:
                print('new ssq: ', ssq)
                print('improvement: ', prev_ssq - ssq)
            # Check if change in objective function is significant
            if (prev_ssq - ssq) < threshold:
                nochange += 1
            self.clusters = new_clusters
            if self.verbose:
                print('End of iteration: ', iteration)
            iteration += 1

        print('tabu_list: ', tabu_list)
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
    # print(X)
    threshold = 10
    Y = tsne.fit_transform(scale(X))
    heu = KMeans(X, 1, 2, True)
    # print("LLOYD HEURISTIC")
    # c, ssq = heu.lloyd_local_search(heu.lloyd_initial_heuristic(), threshold)
    # # print(le.fit_transform(c))
    # plot_clustering_results(Y, le.fit_transform(c),
    #                         'LLOYD', label, 'lloyd')
    # print('Sum of squares:: ', ssq)
    # print('MACQUEEN HEURISTIC')
    # d, ssq = heu.macqueen_heuristic()
    # # print(d)
    # plot_clustering_results(Y, le.fit_transform(d),
    #                         'MACQUEEN', label)
    # print('Sum of squares:: ', ssq)
    # print('K FURTHEST HEURISTIC')
    # e, ssq = heu.k_furthest_initial_heuristic()
    # # print(e)
    # plot_clustering_results(Y, le.fit_transform(e), 'K-FURTHEST',
    #                         label)
    # print('Sum of squares:: ', ssq)
    # print('K POPULAR HEURISTIC')
    # f, ssq = heu.k_popular_initial_heuristic()
    # plot_clustering_results(Y, le.fit_transform(f), 'K-POPULAR',
    #                         label)
    # print('Sum of squares:: ', ssq)
    # print('K-MEANS++ HEURISTIC')
    # f, ssq = heu.lloyd_local_search(heu.k_means_plus_plus(), threshold)
    # plot_clustering_results(Y, le.fit_transform(f),
    #                         'K-MEANS++', label, 'kmeans++')
    # print('Sum of squares:: ', ssq)

    # METAHEURISTICS
    print('Tabu Search METAHEURISTIC')
    f, ssq = heu.tabu_search_metaheuristic(threshold)
    plot_clustering_results(Y, le.fit_transform(f),
                            'K-MEANS++', label, 'kmeans++')
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
