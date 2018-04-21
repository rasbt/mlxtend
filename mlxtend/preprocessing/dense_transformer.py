# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# A class for transforming sparse numpy arrays into dense arrays.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from sklearn.base import BaseEstimator
from scipy.sparse import issparse


class DenseTransformer(BaseEstimator):
    """
    Convert a sparse array into a dense array.

    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/DenseTransformer/

    """

    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        """ Return a dense version of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_dense : dense version of the input X array.

        """
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

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
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        """ Return a dense version of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_dense : dense version of the input X array.

        """
        return self.transform(X=X, y=y)
