# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A class to apply column-based mean centering to a dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .transformer import TransformerObj


class MeanCenterer(TransformerObj):
    """Column centering of vectors and matrices.

    Attributes
    -----------
    col_means : numpy.ndarray [n_columns]
        NumPy array storing the mean values for centering after fitting
        the MeanCenterer object.
    """
    def __init__(self):
        self.col_means = None

    def transform(self, X):
        X_tr = np.copy(self._get_array(X))
        X_tr = np.apply_along_axis(func1d=lambda x: x - self.col_means, axis=1, arr=X_tr)
        return X_tr

    def fit(self, X):
        self.col_means = self._get_array(X).mean(axis=0)
        return self

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _get_array(self, X):
        if isinstance(X, list):
            X_fl = np.asarray(X, dtype='float')[:, None]
        else:
            X_fl = X.astype('float')
        return X_fl
