# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.data import three_blobs_data
from mlxtend.cluster import Kmeans
from mlxtend.utils import assert_raises
import numpy as np


X, y = three_blobs_data()


def test_nonfitted():
    km = Kmeans(k=3,
                max_iter=50,
                random_seed=1,
                print_progress=0)

    assert_raises(AttributeError,
                  'Model is not fitted, yet.',
                  km.predict,
                  X)


def test_three_blobs_1():
    km = Kmeans(k=3,
                max_iter=50,
                random_seed=1,
                print_progress=0)
    y_pred = km.fit(X).predict(X)
    assert (y_pred == y).all()


def test_three_blobs_2():
    km = Kmeans(k=3,
                max_iter=50,
                random_seed=1,
                print_progress=0)

    centroids = np.array([[-1.5947298, 2.92236966],
                          [2.06521743, 0.96137409],
                          [0.9329651, 4.35420713]])

    km.fit(X)
    np.testing.assert_almost_equal(centroids, km.centroids_, decimal=2)
