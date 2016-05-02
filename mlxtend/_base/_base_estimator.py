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


class _BaseEstimator(object):

    """Parent Class Estimator

    A base class that is implemented by
    classifiers, regressors, and clustering estimators.

    """
    def __init__(self, print_progress=0,
                 random_seed=None):
        self.print_progress = print_progress
        self.random_seed = random_seed
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._is_fitted = False
        self._allowed_labels = None

    def _fit(self, X, y=None, init_params=True):
        # Implemented in child class
        pass

    def fit(self, X, y=None, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_params : bool (default: True)
            Re-initializes model parametersprior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X, y=y)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        if init_params:
            self._init_params
        self._fit(X=X, y=y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict targets from X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        target_values : array-like, shape = [n_samples]
          Predicted target values.

        """
        self._check_arrays(X=X)
        if not self._is_fitted:
            raise AttributeError('Model is not fitted, yet.')
        return self._predict(X)

    def _predict(self, X):
        # Implemented in child class
        pass

    def _shuffle_arrays(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

    def _print_progress(self, iteration, n_iter,
                        cost=None, time_interval=10):
        if self.print_progress > 0:
            s = '\rIteration: %d/%d' % (iteration, n_iter)
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
                                   n_iter - ela_sec)
                        if eta_sec < 0.0:
                            eta_sec = 0.0
                        self.eta_str_ = self._to_hhmmss(eta_sec)
                    s += ' | ETA: %s' % self.eta_str_
            stderr.write(s)
            stderr.flush()

    def _to_hhmmss(self, sec):
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return "%d:%02d:%02d" % (h, m, s)

    def _check_arrays(self, X, y=None):
        if isinstance(X, list):
            raise ValueError('X must be a numpy array')
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:, numpy.newaxis]')
        try:
            if y is None:
                return
        except(AttributeError):
            if not len(y.shape) == 1:
                raise ValueError('y must be a 1D array.')

        if not len(y) == X.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

    def _init_params(self, weights_shape, bias_shape=(1,), dtype='float64',
                     scale=0.01, bias_const=0.0, random_seed=None):
        """Initialize weight coefficients."""
        if random_seed:
            np.random.seed(random_seed)
        w = np.random.normal(loc=0.0, scale=scale, size=weights_shape)
        b = np.zeros(shape=bias_shape)
        if bias_const != 0.0:
            b += bias_const
        return b.astype(dtype), w.astype(dtype)

    def _yield_minibatches_idx(self, n_batches, data_ary, shuffle=True):
            indices = np.arange(data_ary.shape[0])

            if shuffle:
                indices = np.random.permutation(indices)
            if n_batches > 1:
                remainder = data_ary.shape[0] % n_batches

                if remainder:
                    minis = np.array_split(indices[:-remainder], n_batches)
                    minis[-1] = np.concatenate((minis[-1],
                                                indices[-remainder:]),
                                               axis=0)
                else:
                    minis = np.array_split(indices, n_batches)

            else:
                minis = (indices,)

            for idx_batch in minis:
                yield idx_batch
