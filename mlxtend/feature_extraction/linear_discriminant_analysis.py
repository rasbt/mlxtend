# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Linear Discriminant Analysis for dimensionality reduction
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .._base import _BaseModel


class LinearDiscriminantAnalysis(_BaseModel):
    """
    Linear Discriminant Analysis Class

    Parameters
    ----------
    n_discriminants : int (default: None)
        The number of discrimants for transformation.
        Keeps the original dimensions of the dataset if `None`.

    Attributes
    ----------
    w_ : array-like, shape=[n_features, n_discriminants]
        Projection matrix
    e_vals_ : array-like, shape=[n_features]
        Eigenvalues in sorted order.
    e_vecs_ : array-like, shape=[n_features]
       Eigenvectors in sorted order.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/

    """
    def __init__(self, n_discriminants=None):
        if n_discriminants is not None and n_discriminants < 1:
            raise AttributeError('n_discriminants must be > 1 or None')
        self.n_discriminants = n_discriminants

    def fit(self, X, y, n_classes=None):
        """ Fit the LDA model with X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        n_classes : int (default: None)
            A positive integer to declare the number of class labels
            if not all class labels are present in a partial training set.
            Gets the number of class labels automatically if None.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X, y=y)
        self._fit(X=X, y=y, n_classes=n_classes)
        self._is_fitted = True
        return self

    def _fit(self, X, y, n_classes=None):

        if self.n_discriminants is None or self.n_discriminants > X.shape[1]:
            n_discriminants = X.shape[1]
        else:
            n_discriminants = self.n_discriminants

        if n_classes:
            self._n_classes = n_classes
        else:
            self._n_classes = np.max(y) + 1
        self._n_features = X.shape[1]

        mean_vecs = self._mean_vectors(X=X, y=y, n_classes=self._n_classes)
        within_scatter = self._within_scatter(X=X,
                                              y=y,
                                              n_classes=self._n_classes,
                                              mean_vectors=mean_vecs)
        between_scatter = self._between_scatter(X=X,
                                                y=y,
                                                mean_vectors=mean_vecs)
        self.e_vals_, self.e_vecs_ = self._eigendecom(
            within_scatter=within_scatter, between_scatter=between_scatter)
        self.w_ = self._projection_matrix(eig_vals=self.e_vals_,
                                          eig_vecs=self.e_vecs_,
                                          n_discriminants=n_discriminants)
        return self

    def transform(self, X):
        """ Apply the linear transformation on X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        X_projected : np.ndarray, shape = [n_samples, n_discriminants]
            Projected training vectors.

        """
        if not hasattr(self, 'w_'):
            raise AttributeError('Object as not been fitted, yet.')
        self._check_arrays(X=X)
        return X.dot(self.w_)

    def _mean_vectors(self, X, y, n_classes):
        mean_vectors = []
        for cl in range(n_classes):
            mean_vectors.append(np.mean(X[y == cl], axis=0))
        return mean_vectors

    def _within_scatter(self, X, y, n_classes, mean_vectors):
        S_W = np.zeros((X.shape[1], X.shape[1]))
        for cl, mv in zip(range(n_classes), mean_vectors):
            class_sc_mat = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == cl]:
                row, mv = row.reshape(X.shape[1], 1), mv.reshape(X.shape[1], 1)
                class_sc_mat += (row - mv).dot((row - mv).T)
            S_W += class_sc_mat
        return S_W

    def _between_scatter(self, X, y, mean_vectors):
        overall_mean = np.mean(X, axis=0)
        S_B = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == i + 1, :].shape[0]
            mean_vec = mean_vec.reshape(X.shape[1], 1)
            overall_mean = overall_mean.reshape(X.shape[1], 1)
            S_B += n * (mean_vec - overall_mean).dot(
                (mean_vec - overall_mean).T)
        return S_B

    def _eigendecom(self, within_scatter, between_scatter):
        e_vals, e_vecs = np.linalg.eig(np.linalg.inv(within_scatter).dot(
            between_scatter))
        sort_idx = np.argsort(e_vals)[::-1]
        e_vals, e_vecs = e_vals[sort_idx], e_vecs[sort_idx]
        return e_vals, e_vecs

    def _projection_matrix(self, eig_vals, eig_vecs, n_discriminants):
        matrix_w = np.vstack([eig_vecs[:, i] for
                              i in range(n_discriminants)]).T
        return matrix_w
