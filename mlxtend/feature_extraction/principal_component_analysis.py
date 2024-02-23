# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Principal Component Analysis for dimensionality reduction.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy as sp

from .._base import _BaseModel


class PrincipalComponentAnalysis(_BaseModel):
    """
    Principal Component Analysis Class

    Parameters
    ----------
    n_components : int (default: None)
        The number of principal components for transformation.
        Keeps the original dimensions of the dataset if `None`.
    solver : str (default: 'svd')
        Method for performing the matrix decomposition.
        {'eigen', 'svd'}
    whitening : bool (default: False)
        Performs whitening such that the covariance matrix of
        the transformed data will be the identity matrix.

    Attributes
    ----------
    w_ : array-like, shape=[n_features, n_components]
        Projection matrix
    e_vals_ : array-like, shape=[n_features]
        Eigenvalues in sorted order.
    e_vecs_ : array-like, shape=[n_features]
       Eigenvectors in sorted order.
    e_vals_normalized_ :  array-like, shape=[n_features]
        Normalized eigen values such that they sum up to 1.
        This is equal to what's often referred to as
        "explained variance ratios."
    loadings_ : array_like, shape=[n_features, n_features]
       The factor loadings of the original variables onto
       the principal components. The columns are the principal
       components, and the rows are the features loadings.
       For instance, the first column contains the loadings onto
       the first principal component. Note that the signs may
       be flipped depending on whether you use the 'eigen' or
       'svd' solver; this does not affect the interpretation
       of the loadings though.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/feature_extraction/PrincipalComponentAnalysis/

    """

    def __init__(self, n_components=None, solver="svd", whitening=False):
        valid_solver = {"eigen", "svd"}
        if solver not in valid_solver:
            raise AttributeError(f"Must be in {valid_solver}. Found {solver}")
        self.solver = solver

        if n_components is not None and n_components < 1:
            raise AttributeError("n_components must be > 1 or None")
        self.n_components = n_components
        self._is_fitted = False
        self.whitening = whitening

    def fit(self, X, y=None):
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
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if self.n_components is None or self.n_components > n_features:
            n_components = n_features
        else:
            n_components = self.n_components

        if self.solver == "eigen":
            cov_mat = self._covariance_matrix(X)
            self.e_vals_, self.e_vecs_ = self._decomposition(cov_mat, n_samples)
        elif self.solver == "svd":
            self.e_vals_, self.e_vecs_ = self._decomposition(X, n_samples)

        self.w_ = self._projection_matrix(
            eig_vals=self.e_vals_,
            eig_vecs=self.e_vecs_,
            whitening=self.whitening,
            n_components=n_components,
        )

        tot = np.sum(self.e_vals_)
        self.e_vals_normalized_ = np.array(
            [(i / tot) for i in sorted(self.e_vals_, reverse=True)]
        )
        self.loadings_ = self._loadings()
        return self

    def transform(self, X):
        """Apply the linear transformation on X.

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
        if not hasattr(self, "w_"):
            raise AttributeError("Object as not been fitted, yet.")

        transformed = X.dot(self.w_)
        if self.whitening:
            # ## Debug 1
            # norm = np.sqrt(np.diag(self.e_vals_[:self.w_.shape[1]]))
            # for i, column in enumerate(transformed.T):
            #    transformed[:, i] /= np.max(norm[:, i])
            norm = np.diag((1.0 / np.sqrt(self.e_vals_[: self.w_.shape[1]])))
            transformed = norm.dot(transformed.T).T
        return transformed

    def _covariance_matrix(self, X):
        mean_vec = np.mean(X, axis=0)
        cov_mat = (X - mean_vec).T.dot((X - mean_vec)) / (X.shape[0] - 1)
        return cov_mat

    def _decomposition(self, mat, n_samples):
        if self.solver == "eigen":
            e_vals, e_vecs = sp.linalg.eig(mat)
        elif self.solver == "svd":
            # Only SVD on mean centered data is equivalent to
            # PCA via covariance matrix (note that computing
            # the covariance matrix will implicitly center
            # the data)
            mat_centered = mat - mat.mean(axis=0)
            u, s, _ = sp.linalg.svd(mat_centered.T, lapack_driver="gesvd")
            e_vecs, e_vals = u, s
            e_vals = e_vals**2 / (n_samples - 1)
            if e_vals.shape[0] < e_vecs.shape[1]:
                new_e_vals = np.zeros(e_vecs.shape[1])
                new_e_vals[: e_vals.shape[0]] = e_vals
                e_vals = new_e_vals

        sort_idx = np.argsort(e_vals)[::-1]
        e_vals, e_vecs = e_vals[sort_idx], e_vecs[:, sort_idx]
        return e_vals, e_vecs

    def _loadings(self):
        """Compute factor loadings"""
        return self.e_vecs_ * np.sqrt(self.e_vals_)

    def _projection_matrix(self, eig_vals, eig_vecs, whitening, n_components):
        matrix_w = np.vstack([eig_vecs[:, i] for i in range(n_components)]).T
        return matrix_w
