# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Estimator for Linear Regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .base import _BaseCluster
import numpy as np
from time import time
from scipy.spatial.distance import euclidean


class Kmeans(_BaseCluster):
    """ K-means clustering class.

    Added in 0.4.1dev

    Parameters
    ------------
    k : int
        Number of clusters
    max_iter : int (default: 10)
        Number of iterations during cluster assignment.
        Cluster re-assignment stops automatically when the algorithm
        converged.
    random_seed : int (default: None)
        Set random state for the initial centroid assignment.
    print_progress : int (default: 0)
        Prints progress in fitting to stderr.
        0: No output
        1: Iterations elapsed
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion

    Attributes
    -----------
    centroids_ : 2d-array, shape={k, n_features}
        Feature values of the k cluster centroids.
    custers_ : dictionary
        The cluster assignments stored as a Python dictionary;
        the dictionary keys denote the cluster indeces and the items are
        Python lists of the sample indices that were assigned to each
        cluster.
    iterations_ : int
        Number of iterations until convergence.

    """

    def __init__(self, k, max_iter=10, random_seed=None, print_progress=0):
        super(Kmeans, self).__init__(print_progress=print_progress,
                                     random_seed=random_seed)
        self.k = k
        self.max_iter = max_iter

    def _fit(self, X):
        """Learn cluster centroids from training data.

        Called in self.fit

        """
        self.iterations_ = 0
        n_samples = X.shape[0]

        # initialize centroids
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids_ = X[idx]

        for _ in range(self.max_iter):

            # assign samples to cluster centroids
            self.clusters_ = {i: [] for i in range(self.k)}
            for sample_idx, cluster_idx in enumerate(
                    self._get_cluster_idx(X=X, centroids=self.centroids_)):
                self.clusters_[cluster_idx].append(sample_idx)

            # recompute centroids
            new_centroids = np.array([np.mean(X[self.clusters_[k]], axis=0)
                                      for k in sorted(self.clusters_.keys())])

            # stop if cluster assignment doesn't change
            if (self.centroids_ == new_centroids).all():
                break
            else:
                self.centroids_ = new_centroids

            self.iterations_ += 1

        return self

    def _get_cluster_idx(self, X, centroids):
        for sample_idx, sample in enumerate(X):
            # dist = [euclidean(sample, c) for c in self.centroids_]
            dist = np.sqrt(np.sum(np.square(sample - self.centroids_), axis=1))
            yield np.argmin(dist)

    def _predict(self, X):
        """Predict cluster labels of X.

        Called in self.predict

        """
        pred = np.array([idx for idx in self._get_cluster_idx(X=X,
                         centroids=self.centroids_)])
        return pred
