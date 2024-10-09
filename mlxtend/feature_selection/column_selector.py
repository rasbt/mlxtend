# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
from sklearn.base import BaseEstimator


class ColumnSelector(BaseEstimator):
    """Object for selecting specific columns from a data set.

    Parameters
    ----------
    cols : array-like (default: None)
        A list specifying the feature indices to be selected. For example,
        [1, 4, 5] to select the 2nd, 5th, and 6th feature columns, and
        ['A','C','D'] to select the name of feature columns A, C and D.
        If None, returns all columns in the array.

    drop_axis : bool (default=False)
        Drops last axis if True and the only one column is selected. This
        is useful, e.g., when the ColumnSelector is used for selecting
        only one column and the resulting array should be fed to e.g.,
        a scikit-learn column selector. E.g., instead of returning an
        array with shape (n_samples, 1), drop_axis=True will return an
        aray with shape (n_samples,).

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/

    """

    def __init__(self, cols=None, drop_axis=False):
        self.cols = cols
        self.drop_axis = drop_axis

    def fit_transform(self, X, y=None):
        """Return a slice of the input array.

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
        """Return a slice of the input array.

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

        # We use the loc or iloc accessor if the input is a pandas dataframe
        if hasattr(X, "loc") or hasattr(X, "iloc"):
            if isinstance(self.cols, tuple):
                self.cols = list(self.cols)
            types = {type(i) for i in self.cols}
            if len(types) > 1:
                raise ValueError(
                    "Elements in `cols` should be all of the same data type."
                )
            if isinstance(self.cols[0], int):
                t = X.iloc[:, self.cols].values
            elif isinstance(self.cols[0], str):
                t = X.loc[:, self.cols].values
            else:
                raise ValueError("Elements in `cols` should be either `int` or `str`.")
        else:
            t = X[:, self.cols]

        if t.shape[-1] == 1 and self.drop_axis:
            t = t.reshape(-1)
        if len(t.shape) == 1 and not self.drop_axis:
            t = t[:, np.newaxis]
        return t

    def fit(self, X, y=None):
        """Mock method. Does nothing.

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
