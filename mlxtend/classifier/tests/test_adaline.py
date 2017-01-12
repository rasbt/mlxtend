# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys
import numpy as np
from mlxtend.classifier import Adaline
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


# Iris Data
X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X = X[0:100]  # class 0 and class 1
y1 = y[0:100]

# standardize
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


def test_invalid_labels_1():
    y2 = np.where(y1 == 0, 2, 1)
    ada = Adaline(epochs=15, eta=0.01, random_seed=1)

    if sys.version_info >= (3, 0):
        objtype = '{(0, 1)}'
    else:
        objtype = 'set([(0, 1)])'

    expect = 'Labels not in %s.\nFound (1, 2)' % objtype

    assert_raises(AttributeError,
                  expect,
                  ada.fit,
                  X,
                  y2,
                  {(0, 1)})


def test_invalid_labels_2():
    y2 = np.where(y1 == 0, -1, 1)
    ada = Adaline(epochs=15, eta=0.01, random_seed=1)
    assert_raises(AttributeError,
                  'y array must not contain negative labels.\nFound [-1  1]',
                  ada.fit,
                  X,
                  y2,
                  {(-1, 1)})


def test_normal_equation():
    t1 = np.array([[-0.08], [1.02]])
    b1 = np.array([0.00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=None,
                  random_seed=None)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, decimal=2)
    np.testing.assert_almost_equal(ada.b_, b1, decimal=2)
    assert (y1 == ada.predict(X_std)).all(), ada.predict(X_std)


def test_gradient_descent():
    t1 = np.array([[-0.08], [1.02]])
    b1 = np.array([0.00])
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1)
    np.testing.assert_almost_equal(ada.w_, t1, decimal=2)
    np.testing.assert_almost_equal(ada.b_, b1, decimal=2)
    assert((y1 == ada.predict(X_std)).all())


def test_score_function():
    ada = Adaline(epochs=30,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1)
    acc = ada.score(X_std, y1)
    assert acc == 1.0, acc


def test_refit_weights():
    t1 = np.array([[-0.08], [1.02]])
    ada = Adaline(epochs=15,
                  eta=0.01,
                  minibatches=1,
                  random_seed=1)
    ada.fit(X_std, y1, init_params=True)
    ada.fit(X_std, y1, init_params=False)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y1 == ada.predict(X_std)).all())


def test_stochastic_gradient_descent():
    t1 = np.array([[-0.08], [1.02]])
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
