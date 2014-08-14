# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# scikit-learn utilities for feature selection

import numpy as np

class FeatureSelector(object):
    """ A feature selector for scikit-learn's Pipeline class that returns
        specified columns from a numpy array.
    
    """
    
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, X):
        col_list = []
        for c in self.cols:
            col_list.append(X[:, c:c+1])
        return np.concatenate(col_list, axis=1)

    def fit(self, X, y=None):
        return self
