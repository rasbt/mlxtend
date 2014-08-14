# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# scikit-learn utilities for feature selection

class ColumnSelector(object):
    """ A feature selector for scikit-learn's Pipeline class that returns
        specified columns from a numpy array.
    
    """
    
    def __init__(self, cols):
        self.cols = cols
        
    def transform(self, X, y=None):
        return X[:, self.cols]

    def fit(self, X, y=None):
        return self
