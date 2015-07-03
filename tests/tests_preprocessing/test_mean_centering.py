import numpy as np
from mlxtend.preprocessing import MeanCenterer


def test_mean_centering():
    X1 = np.array([[1, 2, 3], [4, 5, 6]])
    X1_out = np.array([[-1, -1, -1], [1,  1,  1]])
    mc = MeanCenterer()
    assert(mc.fit_transform(X1).all() == X1_out.all())

    X2 = [1, 2, 3]
    X2_out = np.array([-1, 0, 1])
    mc = MeanCenterer()
    assert(mc.fit_transform(X2).all().all() == X2_out.all())
