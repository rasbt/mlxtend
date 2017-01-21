# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlxtend.tf_regressor import TfLinearRegression
import numpy as np
from numpy.testing import assert_almost_equal


np.random.seed(1)
X = np.array([np.random.normal(1.0, 4.55) for i in range(100)])
y = np.array([x1 * 0.1 + 0.1 + np.random.normal(0.0, 0.05) for x1 in X])
X = X[:, np.newaxis]
X2 = np.hstack((X, X))


def test_univariate_univariate_gradient_descent():
    gd_lr = TfLinearRegression(eta=0.05,
                               epochs=55,
                               random_seed=1,
                               print_progress=0)
    gd_lr.fit(X, y)
    assert_almost_equal(gd_lr.b_, np.array([0.11]), decimal=2)
    assert_almost_equal(gd_lr.w_, np.array([0.10]), decimal=2)
    assert_almost_equal(gd_lr.predict(X), y, decimal=1)


def test_multivariate_gradient_descent():
    gd_lr = TfLinearRegression(eta=0.005,
                               epochs=250,
                               random_seed=1,
                               print_progress=0)
    gd_lr.fit(X2, y)
    assert_almost_equal(gd_lr.predict(X2), y, decimal=1)
    assert_almost_equal(gd_lr.b_, np.array([0.1]), decimal=2)
    assert_almost_equal(gd_lr.w_, np.array([-1.1, 1.2]), decimal=2)


def test_continue_training():
    gd_lr = TfLinearRegression(eta=0.005,
                               epochs=120,
                               random_seed=1,
                               print_progress=0)
    gd_lr.fit(X2, y)
    gd_lr.fit(X2, y, init_params=False)
    assert_almost_equal(gd_lr.predict(X2), y, decimal=1)
    assert_almost_equal(gd_lr.b_, np.array([0.1]), decimal=2)
    assert_almost_equal(gd_lr.w_, np.array([-1.1, 1.2]), decimal=2)
