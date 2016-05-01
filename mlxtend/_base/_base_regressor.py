# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from ._base_supervised_estimator import _BaseSupervisedEstimator


class _BaseRegressor(_BaseSupervisedEstimator):

    """Parent Class Classifier

    A base class that is implemented by regressors

    """
    def __init__(self, print_progress=0, random_seed=0):
        super(_BaseRegressor, self).__init__(
            print_progress=print_progress,
            random_seed=random_seed)

    def _check_target_array(self, y, allowed=None):
        if not np.issubdtype(y[0], float):
            raise AttributeError('y must be a float array.\nFound %s'
                                 % y.dtype)
