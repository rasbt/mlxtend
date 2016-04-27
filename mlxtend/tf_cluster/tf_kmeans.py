# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Estimator for Linear Regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Estimator for Linear Regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.tf_cluster.tf_base import _TfBaseCluster
import tensorflow as tf
import numpy as np
from time import time


class TfKmeans(_TfBaseCluster):
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
                 random_seed=None, print_progress=0, dtype=None):
        super(TfKmeans, self).__init__(print_progress=print_progress,
                                       random_seed=random_seed,
                                       dtype=dtype)

        self.k = k
        self.max_iter = max_iter

    def _fit(self, X):
        """Learn cluster centroids from training data.

        Called in self.fit

        """
        self.iterations_ = 0
        n_samples = X.shape[0]
        n_features = X.shape[1]

        # initialize centroids
        idx = np.random.choice(n_samples, self.k, replace=False)
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

            train_init = tf.initialize_all_variables()

        tf.reset_default_graph()

        sess = tf.Session(graph=self.g_train)
        with sess:
            sess.run(train_init)
            self.init_time_ = time()
            for _ in range(self.max_iter):

                idx, new_centroids = sess.run([assignments, centroids],
                                              feed_dict={tf_X: X,
                                              tf_centroids: self.centroids_})

                if (self.centroids_ == new_centroids).all():
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
            pred_init = tf.initialize_all_variables()

        tf.reset_default_graph()

        sess = tf.Session(graph=self.g_predict)
        with sess:
            sess.run(pred_init)
            pred = sess.run([assignments], feed_dict={tf_X: X,
                            tf_centroids: self.centroids_})[0]

        return pred
