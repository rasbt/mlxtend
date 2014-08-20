# Sebastian Raschka 08/20/2014
# mlxtend Machine Learning Library Extensions

import numpy as np
from .transformer import TransformerObj

class MeanCenterer(TransformerObj):
    """
    Class for column centering of vectors and matrices.
    
    Keyword arguments:
        X: NumPy array object where each attribute/variable is
           stored in an individual column. 
           Also accepts 1-dimensional Python list objects.
    
    Class methods:
        fit: Fits column means to MeanCenterer object.
        transform: Uses column means from `fit` for mean centering.
        fit_transform: Fits column means and performs mean centering.
    
    The class methods `transform` and `fit_transform` return a new numpy array
    object where the attributes are centered at the column means.
    
    """
    def __init__(self):
        self.col_means = None

    def transform(self, X):
        self._get_array(X)
        # centering
        for i in range(self.ary.shape[0]):
            self.ary[i] -= self.col_means
        return self.ary
    
    def fit(self, X):
        self._get_array(X)
        self.col_means = self.ary.mean(axis=0)
        return self
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
