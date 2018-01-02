# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Base Classifier (Classifier Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class _BaseFeatureExtractor(object):
    """Parent Class for Feature Extractors"""

    def __init__(self):
        pass

    def _check_arrays(self, X, y=None):
        if isinstance(X, list):
            raise ValueError('X must be a numpy array')
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:,numpy.newaxis]')
        try:
            if y is None:
                return
        except(AttributeError):
            if not len(y.shape) == 1:
                raise ValueError('y must be a 1D array.')

        if not isinstance(y, np.ndarray):
            raise ValueError('y must be a numpy array')

        if y.shape[0] != X.shape[0]:
            raise ValueError('X and y must contain the same number of samples')
