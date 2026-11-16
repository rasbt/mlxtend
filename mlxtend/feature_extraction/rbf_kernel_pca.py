# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Principal Component Analysis for dimensionality reduction.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.spatial import distance

from .._base import _BaseModel


class RBFKernelPCA(_BaseModel):
    """
    RBF Kernel Principal Component Analysis for dimensionality reduction.

    Parameters
    ----------
    gamma : float (default: 15.0)
        Free parameter (coefficient) of the RBF kernel.
    n_components : int (default: None)
        The number of principal components for transformation.
        Keeps the original dimensions of the dataset if `None`. The fitted
        value is capped at the number of input features.
    copy_X : bool (default: True)
        Copies training data, which is required to compute the projection
        of new data via the transform method. Uses a reference to X if False.

    Attributes
    ----------
    n_components_ : int
        Number of components used for the projection. This is the requested
        `n_components`, capped at the number of input features.
    e_vals_ : array-like, shape=[n_samples]
        Eigenvalues in sorted order.
    e_vecs_ : array-like, shape=[n_samples, n_samples]
       Eigenvectors in sorted order.
    X_projected_ : array-like, shape=[n_samples, n_components_]
       Training samples projected along the component axes.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/feature_extraction/RBFKernelPCA/

    """

    def __init__(self, gamma=15.0, n_components=None, copy_X=True):
        if n_components is not None and n_components < 1:
            raise AttributeError("n_components must be > 1 or None")
        self.n_components = n_components
        self.gamma = gamma
        self.copy_X = copy_X
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
        self.n_components_ = n_components

        kernel_mat = self._kernel_matrix(X)
        self._K_fit_col_means_ = kernel_mat.mean(axis=0)
        self._K_fit_all_mean_ = kernel_mat.mean()
        kernel_mat_centered = self._center_kernel(
            kernel_mat,
            row_means=self._K_fit_col_means_[:, np.newaxis],
            col_means=self._K_fit_col_means_[np.newaxis, :],
            all_mean=self._K_fit_all_mean_,
        )
        self.e_vals_, self.e_vecs_ = self._eigendecom(kernel_mat_centered)
        self.X_projected_ = self._projection_matrix(
            eig_vecs=self.e_vecs_, n_components=n_components
        )
        if self.copy_X:
            self.X_ = X.copy()
        else:
            self.X_ = X
        return self

    def transform(self, X):
        """Apply the non-linear transformation on X.

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
        if not hasattr(self, "X_"):
            raise AttributeError("Object has not been fitted, yet.")
        self._check_arrays(X=X)

        kernel_mat = self._kernel_matrix(X, self.X_)
        kernel_mat_centered = self._center_kernel(
            kernel_mat,
            row_means=kernel_mat.mean(axis=1, keepdims=True),
            col_means=self._K_fit_col_means_[np.newaxis, :],
            all_mean=self._K_fit_all_mean_,
        )

        e_vecs = self._projection_matrix(
            eig_vecs=self.e_vecs_, n_components=self.n_components_
        )
        return kernel_mat_centered.dot(e_vecs / self.e_vals_[: e_vecs.shape[1]])

    @staticmethod
    def _center_kernel(kernel_mat, row_means, col_means, all_mean):
        return kernel_mat - row_means - col_means + all_mean

    def _kernel_matrix(self, X, Y=None):
        if Y is None:
            sq_dists = distance.squareform(distance.pdist(X, "sqeuclidean"))
        else:
            sq_dists = distance.cdist(X, Y, "sqeuclidean")
        return np.exp((-1) * self.gamma * sq_dists)

    def _eigendecom(self, kernel_mat):
        e_vals, e_vecs = np.linalg.eigh(kernel_mat)
        sort_idx = np.argsort(e_vals)[::-1]
        e_vals, e_vecs = e_vals[sort_idx], e_vecs[:, sort_idx]
        return e_vals, e_vecs

    def _projection_matrix(self, eig_vecs, n_components):
        matrix_w = np.vstack([eig_vecs[:, i] for i in range(n_components)]).T
        return matrix_w
