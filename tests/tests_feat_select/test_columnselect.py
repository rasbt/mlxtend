import numpy as np
from mlxtend.sklearn import ColumnSelector

def test_ColumnSelector():
    X1_in = np.ones((4,8))
    X1_out = ColumnSelector(cols=(1,3)).transform(X1_in)
    assert(X1_out.shape == (4,2))

	

