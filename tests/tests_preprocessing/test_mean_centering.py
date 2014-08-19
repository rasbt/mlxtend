import numpy as np
from mlxtend.preprocessing import mean_centering

def test_mean_centering():
    X1 = np.array([[1,2,3], [4,5,6]])
    X1_out = np.array([[-1, -1, -1], [ 1,  1,  1]])
    assert(mean_centering(X1).all() == X1_out.all())

    X2 = [1, 2, 3]
    X2_out = np.array([-1, 0, 1])
    assert(mean_centering(X2).all() == X2_out.all())

