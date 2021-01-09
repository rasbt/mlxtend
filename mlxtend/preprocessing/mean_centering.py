# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A class to apply column-based mean centering to a dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class MeanCenterer(object):

    """Column centering of vectors and matrices.

    Attributes
    -----------
    col_means : numpy.ndarray [n_columns]
        NumPy array storing the mean values for centering after fitting
        the MeanCenterer object.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/MeanCenterer/

    """
    def __init__(self):
        pass

    def transform(self, X):
        """Centers a NumPy array.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Array of data vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        X_tr : {array-like, sparse matrix}, shape = [n_samples, n_features]
            A copy of the input array with the columns centered.

        """
        if not hasattr(self, "col_means"):
            raise AttributeError("MeanCenterer has not been fitted, yet.")
        X_tr = np.copy(self._get_array(X))
        X_tr = np.apply_along_axis(func1d=lambda x: x - self.col_means,
                                   axis=1, arr=X_tr)
        return X_tr

    def fit(self, X):
        """Gets the column means for mean centering.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Array of data vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        self
        """
        self.col_means = self._get_array(X).mean(axis=0)
        return self

    def fit_transform(self, X):
        """Fits and transforms an arry.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Array of data vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        --------
        X_tr : {array-like, sparse matrix}, shape = [n_samples, n_features]
            A copy of the input array with the columns centered.
        """
        self.fit(X)
        return self.transform(X)

    def _get_array(self, X):
        if isinstance(X, list):
            X_fl = np.asarray(X, dtype='float')[:, None]
        else:
            X_fl = X.astype('float')
        return X_fl
