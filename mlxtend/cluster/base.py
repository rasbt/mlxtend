# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sys import stderr
from time import time


class _BaseCluster(object):

    """Parent Class Base Cluster

    A base class that is implemented by
    clustering child classes.

    """
    def __init__(self, print_progress=0, random_seed=None):
        self.print_progress = print_progress
        self.random_seed = random_seed
        self._is_fitted = False

    def fit(self, X):
        """Learn cluster centroids from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_array(X=X)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X)
        self._is_fitted = True
        return self

    def _fit(self, X):
        # Implemented in child class
        pass

    def predict(self, X):
        """Predict cluster labels of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        cluster_labels : array-like, shape = [n_samples]
          Predicted cluster labels.

        """
        self._check_array(X=X)
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def _predict(self, X):
        # Implemented in child class
        pass

    def _shuffle(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

    def _print_progress(self, iteration, cost=None, time_interval=10):
        if self.print_progress > 0:
            s = '\rIteration: %d/%d' % (iteration, self.n_iter)
            if cost:
                s += ' | Cost %.2f' % cost
            if self.print_progress > 1:
                if not hasattr(self, 'ela_str_'):
                    self.ela_str_ = '00:00:00'
                if not iteration % time_interval:
                    ela_sec = time() - self.init_time_
                    self.ela_str_ = self._to_hhmmss(ela_sec)
                s += ' | Elapsed: %s' % self.ela_str_
                if self.print_progress > 2:
                    if not hasattr(self, 'eta_str_'):
                        self.eta_str_ = '00:00:00'
                    if not iteration % time_interval:
                        eta_sec = ((ela_sec / float(iteration)) *
                                   self.n_iter - ela_sec)
                        self.eta_str_ = self._to_hhmmss(eta_sec)
                    s += ' | ETA: %s' % self.eta_str_
            stderr.write(s)
            stderr.flush()

    def _to_hhmmss(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _check_array(self, X):
        if isinstance(X, list):
            raise ValueError('X must be a numpy array')
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:, numpy.newaxis]')
