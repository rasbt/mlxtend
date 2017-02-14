# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Principal Component Analysis for dimensionality reduction.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from .._base import _BaseModel


class PrincipalComponentAnalysis(_BaseModel):
    """
    Principal Component Analysis Class

    Parameters
    ----------
    n_components : int (default: None)
        The number of principal components for transformation.
        Keeps the original dimensions of the dataset if `None`.
    solver : str (default: 'eigen')
        Method for performing the matrix decomposition.
        {'eigen', 'svd'}

    Attributes
    ----------
    w_ : array-like, shape=[n_features, n_components]
        Projection matrix
    e_vals_ : array-like, shape=[n_features]
        Eigenvalues in sorted order.
    e_vecs_ : array-like, shape=[n_features]
       Eigenvectors in sorted order.

    """
    def __init__(self, n_components=None, solver='eigen'):
        valid_solver = {'eigen', 'svd'}
        if solver not in valid_solver:
            raise AttributeError('Must be in %s. Found %s'
                                 % (valid_solver, solver))
        self.solver = solver

        if n_components is not None and n_components < 1:
            raise AttributeError('n_components must be > 1 or None')
        self.n_components = n_components
        self._is_fitted = False

    def fit(self, X):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X)
        self._fit(X=X)
        self._is_fitted = True
        return self

    def _fit(self, X):
        if self.n_components is None or self.n_components > X.shape[1]:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        if self.solver == 'eigen':
            cov_mat = self._covariance_matrix(X)
            self.e_vals_, self.e_vecs_ = self._decomposition(cov_mat)
        elif self.solver == 'svd':
            self.e_vals_, self.e_vecs_ = self._decomposition(X)

        self.w_ = self._projection_matrix(eig_vals=self.e_vals_,
                                          eig_vecs=self.e_vecs_,
                                          n_components=n_components)
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
        X_projected : np.ndarray, shape = [n_samples, n_components]
            Projected training vectors.

        """
        self._check_arrays(X=X)
        if not hasattr(self, 'w_'):
            raise AttributeError('Object as not been fitted, yet.')
        return X.dot(self.w_)

    def _covariance_matrix(self, X):
        mean_vec = np.mean(X, axis=0)
        cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
        return cov_mat

    def _decomposition(self, mat):
        if self.solver == 'eigen':
            e_vals, e_vecs = np.linalg.eig(mat)
        elif self.solver == 'svd':
            u, s, v = np.linalg.svd(mat.T)
            e_vecs, e_vals = u, s

        sort_idx = np.argsort(e_vals)[::-1]
        e_vals, e_vecs = e_vals[sort_idx], e_vecs[sort_idx]
        return e_vals, e_vecs

    def _projection_matrix(self, eig_vals, eig_vecs, n_components):
        matrix_w = np.vstack([eig_vecs[:, i] for i in range(n_components)]).T
        return matrix_w
