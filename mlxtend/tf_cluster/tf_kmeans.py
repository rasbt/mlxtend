# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Estimator for Linear Regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import tensorflow as tf
import numpy as np
from time import time
from .._base import _Cluster
from .._base import _BaseModel
from .._base import _IterativeModel


class TfKmeans(_BaseModel, _Cluster, _IterativeModel):
    """ TensorFlow K-means clustering class.

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
    dtype : Array-type (default: None)
        Uses tensorflow.float32 if None.

    Attributes
    -----------
    centroids_ : 2d-array, shape = {k, n_features}
        Feature values of the k cluster centroids.
    custers_ : dictionary
        The cluster assignments stored as a Python dictionary;
        the dictionary keys denote the cluster indeces and the items are
        Python lists of the sample indices that were assigned to each
        cluster.
    iterations_ : int
        Number of iterations until convergence.

    """

    def __init__(self, k, max_iter=10,
                 convergence_tolerance=1e-05,
                 random_seed=None, print_progress=0, dtype=None):

        self.k = k
        self.max_iter = max_iter
        self.convergence_tolerance = convergence_tolerance
        if dtype is None:
            self.dtype = tf.float32
        else:
            self.dtype = dtype
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, init_params=True):
        """Learn cluster centroids from training data.

        Called in self.fit

        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # initialize centroids

        if init_params:
            self.iterations_ = 0
            # initialize centroids
            rgen = np.random.RandomState(self.random_seed)
            idx = rgen.choice(n_samples, self.k, replace=False)
            self.centroids_ = X[idx]

        self.g_train = tf.Graph()

        with self.g_train.as_default():
            tf_X = tf.placeholder(shape=(n_samples, n_features),
                                  dtype=self.dtype,
                                  name='X_data')
            tf_centroids = tf.placeholder(shape=(self.k, n_features),
                                          dtype=self.dtype,
                                          name='centroids')

            distance_matrices = []
            for idx in range(self.k):
                centroid = tf.slice(tf_centroids,
                                    begin=[idx, 0],
                                    size=[1, n_features])
                euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(
                    tf.sub(tf_X, centroid), 2), reduction_indices=1))
                distance_matrices.append(euclid_dist)
            distance_matrices = tf.reshape(tf.concat(0, distance_matrices),
                                           shape=(self.k, n_samples))

            assignments = tf.argmin(distance_matrices, 0)

            centroids = []
            for clust in range(self.k):
                indices = tf.reshape(tf.where(tf.equal(assignments, clust)),
                                     shape=(1, -1))
                cluster_samples = tf.gather(params=tf_X, indices=indices)
                centroid = tf.reduce_mean(cluster_samples,
                                          reduction_indices=[1])
                centroids.append(centroid)
            centroids = tf.concat(0, centroids)

            train_init = tf.global_variables_initializer()

        tf.reset_default_graph()

        sess = tf.Session(graph=self.g_train)
        with sess:
            sess.run(train_init)
            self.init_time_ = time()
            for _ in range(self.max_iter):

                idx, new_centroids = sess.run([assignments, centroids],
                                              feed_dict={tf_X: X,
                                              tf_centroids: self.centroids_})

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

            self.clusters_ = {i: np.arange(n_samples)[idx == i]
                              for i in range(self.k)}

        return self

    def _predict(self, X):
        """Predict cluster labels of X.

        Called in self.predict

        """
        n_samples = X.shape[0]
        n_features = X.shape[1]

        self.g_predict = tf.Graph()

        with self.g_predict.as_default():
            tf_X = tf.placeholder(shape=(n_samples, n_features),
                                  dtype=self.dtype,
                                  name='X_data')
            tf_centroids = tf.placeholder(shape=(self.k, n_features),
                                          dtype=self.dtype,
                                          name='centroids')

            distance_matrices = []
            for idx in range(self.k):
                centroid = tf.slice(tf_centroids,
                                    begin=(idx, 0),
                                    size=(1, n_features))
                euclid_dist = tf.sqrt(tf.reduce_sum(
                    tf.pow(tf.sub(tf_X, centroid), 2), reduction_indices=1))
                distance_matrices.append(euclid_dist)
            distance_matrices = tf.reshape(tf.concat(0, distance_matrices),
                                           shape=(self.k, n_samples))
            assignments = tf.argmin(distance_matrices, 0)
            pred_init = tf.global_variables_initializer()

        tf.reset_default_graph()

        sess = tf.Session(graph=self.g_predict)
        with sess:
            sess.run(pred_init)
            pred = sess.run([assignments], feed_dict={tf_X: X,
                            tf_centroids: self.centroids_})[0]

        return pred
