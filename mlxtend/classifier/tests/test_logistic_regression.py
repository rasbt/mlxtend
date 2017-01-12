# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys
import numpy as np
from mlxtend.classifier import LogisticRegression
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X = X[0:100]  # class 0 and class 1
y = y[0:100]  # class 0 and class 1

# standardize
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


def test_invalid_labels_1():
    y1 = np.where(y == 0, 2, 1)
    lr = LogisticRegression(epochs=15, eta=0.01, random_seed=1)

    if sys.version_info >= (3, 0):
        objtype = '{(0, 1)}'
    else:
        objtype = 'set([(0, 1)])'

    expect = 'Labels not in %s.\nFound (1, 2)' % objtype

    assert_raises(AttributeError,
                  expect,
                  lr.fit,
                  X,
                  y1,
                  {(0, 1)})


def test_invalid_labels_2():
    y1 = np.where(y == 0, -1, 1)
    lr = LogisticRegression(epochs=15, eta=0.01, random_seed=1)
    assert_raises(AttributeError,
                  'y array must not contain negative labels.\nFound [-1  1]',
                  lr.fit,
                  X,
                  y1,
                  {(-1, 1)})


def test_logistic_regression_gd():
    w = np.array([[1.2], [4.4]])
    b = np.array([0.52])
    lr = LogisticRegression(epochs=100,
                            eta=0.01,
                            minibatches=1,
                            random_seed=1)

    lr.fit(X, y)
    np.testing.assert_almost_equal(lr.w_, w, 2)
    np.testing.assert_almost_equal(lr.b_, b, 2)
    y_pred = lr.predict(X)
    acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
    assert acc == 1.0, "Acc: %s" % acc


def test_score_function():
    lr = LogisticRegression(epochs=100,
                            eta=0.01,
                            minibatches=1,
                            random_seed=1)

    lr.fit(X, y)
    acc = lr.score(X, y)
    assert acc == 1.0, "Acc: %s" % acc


def test_refit_weights():
    w = np.array([[1.2], [4.4]])
    b = np.array([0.52])
    lr = LogisticRegression(epochs=50,
                            eta=0.01,
                            minibatches=1,
                            random_seed=1)

    lr.fit(X, y)
    w1 = lr.w_[0][0]
    w2 = lr.w_[0][0]
    lr.fit(X, y, init_params=False)

    assert w1 != lr.w_[0][0]
    assert w2 != lr.w_[1][0]
    np.testing.assert_almost_equal(lr.w_, w, 2)
    np.testing.assert_almost_equal(lr.b_, b, 2)


def test_predict_proba():
    lr = LogisticRegression(epochs=100,
                            eta=0.01,
                            minibatches=1,
                            random_seed=1)

    lr.fit(X, y)
    idx = [0, 48, 99]  # sample labels: 0, 0, 1
    y_pred = lr.predict_proba(X[idx])
    expect = np.array([0.009, 0.012, 0.993])
    np.testing.assert_almost_equal(y_pred, expect, 3)


def test_logistic_regression_sgd():
    w = np.array([[1.18], [4.38]])
    lr = LogisticRegression(epochs=100,
                            eta=0.01,
                            minibatches=len(y),
                            random_seed=1)

    lr.fit(X, y)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, w, 2)
    y_pred = lr.predict(X)
    acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
    assert acc == 1.0, "Acc: %s" % acc


def test_l2_regularization_gd():
    lr = LogisticRegression(eta=0.01,
                            epochs=20,
                            minibatches=1,
                            l2_lambda=1.0,
                            random_seed=1)
    lr.fit(X, y)
    y_pred = lr.predict(X)
    expect_weights = np.array([[1.061], [2.280]])

    np.testing.assert_almost_equal(lr.w_, expect_weights, 3)
    y_pred = lr.predict(X)
    acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
    assert acc == 1.0, "Acc: %s" % acc


def test_l2_regularization_sgd():
    lr = LogisticRegression(eta=0.01,
                            epochs=100,
                            minibatches=len(y),
                            l2_lambda=1.0,
                            random_seed=1)
    lr.fit(X, y)
    y_pred = lr.predict(X)
    expect_weights = np.array([[0.24], [0.35]])

    np.testing.assert_almost_equal(lr.w_, expect_weights, 2)
    y_pred = lr.predict(X)
    acc = np.sum(y == y_pred, axis=0) / float(X.shape[0])
    assert acc == 0.97, "Acc: %s" % acc


def test_ary_persistency_in_shuffling():
    orig = X.copy()
    lr = LogisticRegression(eta=0.01,
                            epochs=100,
                            minibatches=len(y),
                            l2_lambda=1.0,
                            random_seed=1)
    lr.fit(X, y)
    np.testing.assert_almost_equal(orig, X, 6)
