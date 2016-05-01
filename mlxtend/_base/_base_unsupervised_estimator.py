# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from ._base_estimator import _BaseEstimator


class _BaseUnsupervisedEstimator(_BaseEstimator):

    """Parent Class Supervised Estimator

    A base class that is implemented by
    classifiers and regressors

    """
    def __init__(self, print_progress=0, random_seed=0):
        super(_BaseUnsupervisedEstimator, self).__init__(
            print_progress=print_progress,
            random_seed=random_seed)

    def fit(self, X, init_params=True):
        """Learn model from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        init_params : bool (default: True)
            Re-initializes model parametersprior to fitting.
            Set False to continue training with weights from
            a previous model fitting.

        Returns
        -------
        self : object

        """
        self._is_fitted = False
        self._check_arrays(X=X)
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        self._fit(X=X, init_params=init_params)
        self._is_fitted = True
        return self
