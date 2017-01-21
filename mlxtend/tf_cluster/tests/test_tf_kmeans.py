# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.data import three_blobs_data
from mlxtend.tf_cluster import TfKmeans
from mlxtend.utils import assert_raises
import numpy as np


X, y = three_blobs_data()


def test_nonfitted():
    km = TfKmeans(k=3,
                  max_iter=50,
                  random_seed=1,
                  print_progress=0)

    assert_raises(AttributeError,
                  'Model is not fitted, yet.',
                  km.predict,
                  X)


def test_three_blobs_multi():
    km = TfKmeans(k=3,
                  max_iter=50,
                  random_seed=1,
                  print_progress=0)
    y_pred = km.fit(X).predict(X)
    assert (y_pred == y).all()


def test_three_blobs_1sample():
    km = TfKmeans(k=3,
                  max_iter=50,
                  random_seed=1,
                  print_progress=0)
    sample = X[1, :].reshape(1, 2)

    y_pred = km.fit(X).predict(sample)
    assert y_pred[0] == y[1]


def test_three_blobs_centroids():
    km = TfKmeans(k=3,
                  max_iter=50,
                  random_seed=1,
                  print_progress=0)

    centroids = np.array([[-1.5947298, 2.92236966],
                          [2.06521743, 0.96137409],
                          [0.9329651, 4.35420713]])

    km.fit(X)
    np.testing.assert_almost_equal(centroids, km.centroids_, decimal=2)


def test_continue_training():
    km = TfKmeans(k=3,
                  max_iter=1,
                  random_seed=1,
                  print_progress=0)

    first_iter = np.array([[-1.33, 3.26],
                           [1.95, 0.99],
                           [1.09, 4.26]])

    second_iter = np.array([[-1.5947298, 2.92236966],
                            [2.06521743, 0.96137409],
                            [0.9329651, 4.35420713]])

    km.fit(X)
    np.testing.assert_almost_equal(first_iter, km.centroids_, decimal=2)
    assert km.iterations_ == 1, km.iterations_

    km.fit(X, init_params=False)
    np.testing.assert_almost_equal(second_iter, km.centroids_, decimal=2)
    assert km.iterations_ == 2, km.iterations_
