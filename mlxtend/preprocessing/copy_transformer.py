# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A Class that returns a copy of a dataset in a scikit-learn pipeline.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import BaseEstimator
from scipy.sparse import issparse


class CopyTransformer(BaseEstimator):
    """Transformer that returns a copy of the input array

    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/

    """
    def __init__(self):
        pass

    def transform(self, X, y=None):
        """ Return a copy of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_copy : copy of the input X array.

        """
        if isinstance(X, list):
            return np.asarray(X)
        elif isinstance(X, np.ndarray) or issparse(X):
            return X.copy()
        else:
            raise ValueError('X must be a list or NumPy array'
                             ' or SciPy sparse array. Found %s'
                             % type(X))

    def fit_transform(self, X, y=None):
        """ Return a copy of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_copy : copy of the input X array.

        """
        return self.transform(X)

    def fit(self, X, y=None):
        """ Mock method. Does nothing.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        self

        """
        return self
