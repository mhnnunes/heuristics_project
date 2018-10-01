#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sys import argv
from io_utils import read_input
from sklearn.metrics.pairwise import euclidean_distances


class KMeans(object):
    """ This class implements several heuristics for the
    Euclidean Minimum Sum-of-Squares Clustering (MSCC) problem.
    This problem is also commonly known as K-Means.
    """

    def __init__(self, data, seed, k):
        self.k = k
        # Data will be passed as a numpy 2D array
        self.data = data
        self.seed = seed
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
                    np.argmin(euclidean_distances(X=heu.data,
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
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
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
                    np.argmin(euclidean_distances(X=heu.data,
                                                  Y=centroid.reshape(1, -1)))
                # Update points that were already in the cluster that changed
                for p in points_in_cluster:
                    self.clusters[p] = closest_point_index

                # Remove old centroid and add new one to list of centers
                centers_indexes = np.append(centers_indexes,
                                            closest_point_index)
                centers_indexes = np.delete(centers_indexes, closest_center)

            # Check if there were any changes of clusters
            if changed_cluster_prev == changed_cluster_cur:
                nochange += 1
            changed_cluster_prev = changed_cluster_cur
            # print(changed_cluster_cur, ' points changed cluster')
            iteration += 1
        # Calculate objective function value
        self.clusters = self.clusters.astype(int)
        ssq = self.__calculate_sum_of_squares()
        return self.clusters, ssq

    def k_furthest_initial_heuristic(self, threshold=10):
        # If the 'distances' variable does not exist, make it
        if not hasattr(self, 'distances'):
            self.calculate_distance_between_pairs()

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
                    np.argmin(euclidean_distances(X=heu.data,
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
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
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
                    np.argmin(euclidean_distances(X=heu.data,
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
            # No change since last iteration
            if changed_cluster_cur == changed_cluster_prev:
                nochange += 1
            # Save current clusters
            self.clusters = cur_clusters
            # Save current number of points that changed cluster
            changed_cluster_prev = changed_cluster_cur
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
    # pre-process data
    # pre-processing breast cancer data
    data = data.drop('id', axis=1)
    data = data.drop('Unnamed: 32', axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    data = data.drop('diagnosis', axis=1)
    print(data)
    print(data.values)
    heu = KMeans(data.values, 1, 5)
    print("LLOYD HEURISTIC")
    c, ssq = heu.lloyd_heuristic()
    # print(c)
    print('Sum of squares:: ', ssq)
    print('MACQUEEN HEURISTIC')
    d, ssq = heu.macqueen_heuristic()
    # print(d)
    print('Sum of squares:: ', ssq)
    print('K FURTHEST HEURISTIC')
    e, ssq = heu.k_furthest_initial_heuristic()
    # print(e)
    print('Sum of squares:: ', ssq)
    print('K POPULAR HEURISTIC')
    f, ssq = heu.k_popular_initial_heuristic()
    print('Sum of squares:: ', ssq)