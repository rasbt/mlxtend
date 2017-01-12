# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from sklearn.base import BaseEstimator
import numpy as np


class ColumnSelector(BaseEstimator):

    def __init__(self, cols=None):
        """Object for selecting specific columns from a data set.

        Parameters
        ----------
        cols : array-like (default: None)
            A list specifying the feature indices to be selected. For example,
            [1, 4, 5] to select the 2nd, 5th, and 6th feature columns.
            If None, returns all columns in the array.
        """
        self.cols = cols

    def fit_transform(self, X, y=None):
        """ Return a slice of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features

        """
        return self.transform(X=X, y=y)

    def transform(self, X, y=None):
        """ Return a slice of the input array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] (default: None)

        Returns
        ---------
        X_slice : shape = [n_samples, k_features]
            Subset of the feature space where k_features <= n_features

        """
        t = X[:, self.cols]
        if len(t.shape) == 1:
            t = t[:, np.newaxis]
        return t

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
