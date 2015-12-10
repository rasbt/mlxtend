# Sebastian Raschka 08/20/2014
# mlxtend Machine Learning Library Extensions

import numpy as np
from .transformer import TransformerObj


class MeanCenterer(TransformerObj):
    """
    Class for column centering of vectors and matrices.

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
        X_fl = X.astype('float64')
        return X_fl
