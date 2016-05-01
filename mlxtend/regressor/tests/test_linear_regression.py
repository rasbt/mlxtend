# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlxtend.regressor import LinearRegression
from mlxtend.data import boston_housing_data
import numpy as np
from numpy.testing import assert_almost_equal

X, y = boston_housing_data()
X_rm = X[:, 5][:, np.newaxis]
X_rm_lstat = X[:, [5, -1]]

# standardized variables
X_rm_std = (X_rm - X_rm.mean(axis=0)) / X_rm.std(axis=0)
X_rm_lstat_std = ((X_rm_lstat - X_rm_lstat.mean(axis=0)) /
                  X_rm_lstat.std(axis=0))
y_std = (y - y.mean()) / y.std()


def test_univariate_normal_equation():
    w_exp = np.array([[9.102]])
    b_exp = np.array([-34.671])
    ne_lr = LinearRegression(minibatches=None)
    ne_lr.fit(X_rm, y)
    assert_almost_equal(ne_lr.w_, w_exp, decimal=3)
    assert_almost_equal(ne_lr.b_, b_exp, decimal=3)


def test_univariate_normal_equation_std():
    w_exp = np.array([[0.695]])
    b_exp = np.array([0.00])
    ne_lr = LinearRegression(minibatches=None)
    ne_lr.fit(X_rm_std, y_std)
    assert_almost_equal(ne_lr.w_, w_exp, decimal=3)
    assert_almost_equal(ne_lr.b_, b_exp, decimal=3)


def test_univariate_gradient_descent():
    w_exp = np.array([[0.695]])
    b_exp = np.array([0.00])
    gd_lr = LinearRegression(minibatches=1,
                             eta=0.001,
                             epochs=500,
                             random_seed=0)
    gd_lr.fit(X_rm_std, y_std)
    assert_almost_equal(gd_lr.w_, w_exp, decimal=3)
    assert_almost_equal(gd_lr.b_, b_exp, decimal=3)


def test_univariate_stochastic_gradient_descent():
    w_exp = np.array([[0.695]])
    b_exp = np.array([0.00])
    sgd_lr = LinearRegression(minibatches=len(y),
                              eta=0.0001,
                              epochs=150,
                              random_seed=0)
    sgd_lr.fit(X_rm_std, y_std)
    assert_almost_equal(sgd_lr.w_, w_exp, decimal=3)
    assert_almost_equal(sgd_lr.b_, b_exp, decimal=3)


def test_multivariate_normal_equation():
    w_exp = np.array([[5.095], [-0.642]])
    b_exp = np.array([-1.358])
    ne_lr = LinearRegression(minibatches=None)
    ne_lr.fit(X_rm_lstat, y)
    assert_almost_equal(ne_lr.w_, w_exp, decimal=3)
    assert_almost_equal(ne_lr.b_, b_exp, decimal=3)


def test_multivariate_gradient_descent():
    w_exp = np.array([[0.389], [-0.499]])
    b_exp = np.array([0.000])
    gd_lr = LinearRegression(eta=0.001,
                             epochs=500,
                             minibatches=1,
                             random_seed=0)
    gd_lr.fit(X_rm_lstat_std, y_std)
    assert_almost_equal(gd_lr.w_, w_exp, decimal=3)
    assert_almost_equal(gd_lr.b_, b_exp, decimal=3)


def test_multivariate_stochastic_gradient_descent():
    w_exp = np.array([[0.389], [-0.499]])
    b_exp = np.array([0.000])
    sgd_lr = LinearRegression(eta=0.0001,
                              epochs=500,
                              minibatches=len(y),
                              random_seed=0)
    sgd_lr.fit(X_rm_lstat_std, y_std)
    assert_almost_equal(sgd_lr.w_, w_exp, decimal=3)
    assert_almost_equal(sgd_lr.b_, b_exp, decimal=3)


def test_ary_persistency_in_shuffling():
    orig = X_rm_lstat_std.copy()
    sgd_lr = LinearRegression(eta=0.0001,
                              epochs=500,
                              minibatches=len(y),
                              random_seed=0)
    sgd_lr.fit(X_rm_lstat_std, y_std)
    np.testing.assert_almost_equal(orig, X_rm_lstat_std, 6)
