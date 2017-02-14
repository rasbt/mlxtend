# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import iris_data
from mlxtend.utils import assert_raises


X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X_bin = X[0:100]  # class 0 and class 1
y_bin = y[0:100]  # class 0 and class 1

# standardize
X_bin[:, 0] = (X_bin[:, 0] - X_bin[:, 0].mean()) / X_bin[:, 0].std()
X_bin[:, 1] = (X_bin[:, 1] - X_bin[:, 1].mean()) / X_bin[:, 1].std()
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


def test_labels():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([-1, 1])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    assert_raises(AttributeError,
                  'y array must not contain negative labels.\nFound [-1  1]',
                  lr.fit,
                  X,
                  y)


def test_binary_logistic_regression_gd():
    t = np.array([[0.13, -0.12],
                  [-3.07, 3.05]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)

    lr.fit(X_bin, y_bin)
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_refit_weights():
    t = np.array([[0.13, -0.12],
                  [-3.07, 3.05]])
    lr = SoftmaxRegression(epochs=100,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)

    lr.fit(X_bin, y_bin)
    w1 = lr.w_[0][0]
    w2 = lr.w_[0][0]
    lr.fit(X_bin, y_bin, init_params=False)

    assert w1 != lr.w_[0][0]
    assert w2 != lr.w_[1][0]
    np.testing.assert_almost_equal(lr.w_, t, 2)


def test_binary_logistic_regression_sgd():
    t = np.array([[0.13, -0.12],
                  [-3.06, 3.05]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=len(y_bin),
                           random_seed=1)

    lr.fit(X_bin, y_bin)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_binary_l2_regularization_gd():
    t = np.array([[-0.17, 0.17],
                  [-2.26, 2.26]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           l2=1.0,
                           minibatches=1,
                           random_seed=1)

    lr.fit(X_bin, y_bin)
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_multi_logistic_regression_gd_weights():
    t = np.array([[-0.95, -2.45, 3.4],
                  [-3.95, 2.34, 1.59]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    np.testing.assert_almost_equal(lr.w_, t, 2)


def test_multi_logistic_probas():
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    idx = [0, 50, 149]  # sample labels: 0, 1, 2
    y_pred = lr.predict_proba(X[idx])
    exp = np.array([[0.99, 0.01, 0.00],
                    [0.01, 0.88, 0.11],
                    [0.00, 0.02, 0.98]])
    np.testing.assert_almost_equal(y_pred, exp, 2)


def test_multi_logistic_regression_gd_acc():
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    assert (y == lr.predict(X)).all()


def test_score_function():
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    acc = lr.score(X, y)
    assert acc == 1.0, acc
