# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
from mlxtend.preprocessing import shuffle_arrays_unison


def test_shuffle_arrays_unison():
    X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y1 = np.array([1, 2, 3])

    X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)

    assert(X2.all() == np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]).all())
    assert(y2.all() == np.array([2, 1, 3]).all())
