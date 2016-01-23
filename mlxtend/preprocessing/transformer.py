# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A Class that returns a copy of a dataset in a scikit-learn pipeline.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class TransformerObj(object):
    def __init__(self):
        self.ary = None

    def _get_array(self, X):
        if isinstance(X, list):
            self.ary = np.asarray(X)
        else:
            self.ary = np.copy(X)
