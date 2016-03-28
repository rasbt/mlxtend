# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier import Adaline
from mlxtend.data import iris_data
import numpy as np
from nose.tools import raises


# Iris Data
X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X = X[0:100]  # class 0 and class 1
y0 = y[0:100]  # class 0 and class 1
y1 = np.where(y[0:100] == 0, -1, 1)  # class -1 and class 1
y2 = np.where(y[0:100] == 0, -2, 1)  # class -2 and class 1

# standardize
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


@raises(Exception)
def test_array_dimensions():
    ada = Adaline(epochs=15, eta=0.01, random_seed=1)
    ada = ada.fit(np.array([1, 2, 3]), [-1])


def test_normal_equation():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=None,
                  random_seed=1)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_gradient_descent():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_score_function():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1)
    acc = ada.score(X_std, y1)
    assert acc == 1.0, acc


def test_refit_weights():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=15,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1, init_weights=True)
    ada.fit(X_std, y1, init_weights=False)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_standardized_iris_data_with_zero_weights():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1,
                  zero_init_weight=True)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_stochastic_gradient_descent():
    t1 = np.array([-5.21e-16, -7.86e-02, 1.02e+00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=len(y),
                  random_seed=1)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_ary_persistency_in_shuffling():
    orig = X_std.copy()
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=len(y),
                  random_seed=1)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(orig, X_std, 6)


def test_0_1_class():
    t1 = np.array([0.51, -0.04, 0.51])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y0)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y0 == ada.predict(X_std)).all())


def test_invalid_class():
    ada = Adaline(epochs=40, eta=0.01, random_seed=1)
    try:
        ada.fit(X, y2)  # 0, 1 class
    except ValueError:
        pass
