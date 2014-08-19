# Sebastian Raschka 08/19/2014
# mlxtend Machine Learning Library Extensions
# Submodules with preprocessing functions.

from .mean_centering import mean_centering
import numpy as np

def mean_centering(X, copy=True):
    """
    Function that performs column centering.
    Keyword arguments:
        X: NumPy array object where each attribute/variable is
           stored in an individual column. 
           Also accepts 1-dimensional Python list objects.
        copy: Returns a copy of the input array if True, otherwise
              performs operation in-place.
              
    """
    if copy:
        mat = np.copy(X)
    else:
        mat = X
    if isinstance(X, list):
        mat = np.asarray(mat)
        
    # centering
    col_means = mat.mean(axis=0)
    for i in range(mat.shape[0]):
        mat[i] -= col_means
    return mat 
