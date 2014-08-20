# Sebastian Raschka 08/20/2014
# mlxtend Machine Learning Library Extensions

import numpy as np


class TransformerObj(object):
    def __init__(self):
        self.ary = None
        
    def _get_array(self, X):
        if isinstance(X, list):
            self.ary = np.asarray(X)
        else:
            self.ary = np.copy(X)
