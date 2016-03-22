# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Classifier (Classifier Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sys import stderr
from time import time


class _TfBaseClassifier(object):

    """Parent Class Base Classifier

    A base class that is important by
    classifier child classes.

    """
    def __init__(self, print_progress=0):
        self.print_progress = print_progress

    def fit(self, X, y, init_weights=True):
        """Learn weight coefficients from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_weights : bool (default: None)
            Reinitialize weights

        Returns
        -------
        self : object

        """
        if not (init_weights is None or isinstance(init_weights, bool)):
            raise AttributeError("init_weights must be True, False, or None")
        init_weights
        self._check_arrays(X=X, y=y)
        return self

    def predict(self, X):
        """Predict class labels of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class_labels : array-like, shape = [n_samples]
          Predicted class labels.

        """
        self._check_arrays(X)
        return self._predict(X)

    def _predict(X, y):
        # Implemented in child class
        pass

    def _shuffle(self, arrays):
        """Shuffle arrays in unison."""
        r = np.random.permutation(len(arrays[0]))
        return [ary[r] for ary in arrays]

    def _print_progress(self, epoch, cost=None, time_interval=10):
        if self.print_progress > 0:
            s = '\rEpoch: %d/%d' % (epoch, self.epochs)
            if cost:
                s += ' | Cost %.2f' % cost
            if self.print_progress > 1:
                if not hasattr(self, 'ela_str_'):
                    self.ela_str_ = '00:00:00'
                if not epoch % time_interval:
                    ela_sec = time() - self.init_time_
                    self.ela_str_ = self._to_hhmmss(ela_sec)
                s += ' | Elapsed: %s' % self.ela_str_
                if self.print_progress > 2:
                    if not hasattr(self, 'eta_str_'):
                        self.eta_str_ = '00:00:00'
                    if not epoch % time_interval:
                        eta_sec = ((ela_sec / float(epoch)) *
                                   self.epochs - ela_sec)
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
            raise ValueError('X must be a 2D array. Try X[:,numpy.newaxis]')
        try:
            if y is None:
                return
        except(AttributeError):
            if not len(y.shape) == 1:
                raise ValueError('y must be a 1D array.')

        if not len(y) == X.shape[0]:
            raise ValueError('X and y must contain the same number of samples')

    def _tensor_to_numpy(self, var):
        sess = tf.Session()
        with sess.as_default():
            tf.initialize_all_variables().run()
            return var.eval()

    def _one_hot(self, y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(float)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
