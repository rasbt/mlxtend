# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A class for transforming sparse numpy arrays into dense arrays.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from sklearn.base import BaseEstimator
from scipy.sparse import issparse


class DenseTransformer(BaseEstimator):
    """Convert a sparse array into a dense array."""

    def __init__(self, return_copy=True):
        self.return_copy = return_copy
        self.is_fitted = False

    def transform(self, X, y=None):
        if issparse(X):
            return X.toarray()
        elif self.return_copy:
            return X.copy()
        else:
            return X

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X=X, y=y)
