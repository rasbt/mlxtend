# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
from mlxtend.preprocessing import MeanCenterer


def test_mean_centering():
    X1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    X1_out = np.array([[-1.0, -1.0, -1.0], [1.0,  1.0,  1.0]])
    mc = MeanCenterer()
    assert(mc.fit_transform(X1).all() == X1_out.all())

    X2 = [1.0, 2.0, 3.0]
    X2_out = np.array([-1.0, 0.0, 1.0])
    mc = MeanCenterer()
    assert(mc.fit_transform(X2).all().all() == X2_out.all())
