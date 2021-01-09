# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Estimator for Linear Regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
from .._base import _Cluster
from .._base import _BaseModel
from .._base import _IterativeModel
# from scipy.spatial.distance import euclidean


class Kmeans(_BaseModel, _Cluster, _IterativeModel):
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
    convergence_tolerance : float (default: 1e-05)
        Compares current centroids with centroids of the previous iteration
        using the given tolerance (a small positive float)to determine
        if the algorithm converged early.
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

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/Kmeans/

    """

    def __init__(self, k, max_iter=10,
                 convergence_tolerance=1e-05,
                 random_seed=None, print_progress=0):

        _BaseModel.__init__(self)
        _Cluster.__init__(self)
        _IterativeModel.__init__(self)
        self.k = k
        self.max_iter = max_iter
        self.convergence_tolerance = convergence_tolerance
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, init_params=True):
        """Learn cluster centroids from training data.

        Called in self.fit

        """
        n_samples = X.shape[0]

        if init_params:
            self.iterations_ = 0
            # initialize centroids
            rgen = np.random.RandomState(self.random_seed)
            idx = rgen.choice(n_samples, self.k, replace=False)
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

            if np.allclose(self.centroids_, new_centroids,
                           rtol=self.convergence_tolerance,
                           atol=1e-08, equal_nan=False):
                break
            else:
                self.centroids_ = new_centroids

            self.iterations_ += 1
            if self.print_progress:
                self._print_progress(iteration=self.iterations_,
                                     n_iter=self.max_iter)

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
