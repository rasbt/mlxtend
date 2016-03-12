# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier import SoftmaxRegression
from mlxtend.data import iris_data
import numpy as np


X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X_bin = X[0:100]  # class 0 and class 1
y_bin = y[0:100]  # class 0 and class 1

# standardize
X_bin[:, 0] = (X_bin[:, 0] - X_bin[:, 0].mean()) / X_bin[:, 0].std()
X_bin[:, 1] = (X_bin[:, 1] - X_bin[:, 1].mean()) / X_bin[:, 1].std()
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


def test_binary_logistic_regression_gd():
    t = np.array([[1.11, -0.1],
                  [-4.12, 2.52]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)

    lr.fit(X_bin, y_bin)
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y_bin == lr.predict(X_bin)).all())


def test_binary_logistic_regression_sgd():
    t = np.array([[0.56, 0.45],
                  [-4.18, 2.58]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=len(y_bin),
                           random_seed=1)

    lr.fit(X_bin, y_bin)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y_bin == lr.predict(X_bin)).all())


def test_binary_l2_regularization_gd():
    lr = SoftmaxRegression(eta=0.005,
                           epochs=200,
                           minibatches=1,
                           l2_lambda=1.0,
                           random_seed=1)
    lr.fit(X_bin, y_bin)
    y_pred = lr.predict(X_bin)
    expect_weights = np.array([[0.186, 0.186],
                               [-2.625, 2.037]])

    np.testing.assert_almost_equal(lr.w_, expect_weights, 3)
    acc = sum(y_pred == y_bin) / len(y_bin)
    assert(acc == 1.0)


def test_multi_logistic_regression_gd_weights():
    t = np.array([[-0.17, -2.86, 3.51],
                  [-4.85, 2.0, 0.35]])
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    np.testing.assert_almost_equal(lr.w_, t, 2)


def test_multi_logistic_regression_gd_acc():
    lr = SoftmaxRegression(epochs=200,
                           eta=0.005,
                           minibatches=1,
                           random_seed=1)
    lr.fit(X, y)
    assert((y == lr.predict(X)).all())
