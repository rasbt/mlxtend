# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# scikit-learn utilities for transforming a sparse
# numpy array into a dense numpy array

class DenseTransformer(object):
    """ 
    A transformer for scikit-learn's Pipeline class that converts
    a sparse matrix into a dense matrix.
    
    """
    
    def __init__(self, some_param=True):
        pass

    def transform(self, X, y=None):
        return X.toarray()

    def fit(self, X, y=None):
        return self
    
    def fit_transform(self, X, y=None):
        return X.toarray()
    
    def get_params(self, deep=True):
        return {'some_param':True}