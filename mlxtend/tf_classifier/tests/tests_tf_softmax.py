# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.tf_classifier import TfSoftmaxRegression
from mlxtend.data import iris_data
import numpy as np
from nose.tools import raises


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
    t = np.array([[-0.28, 0.95],
                  [-2.23, 2.4]])
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)

    lr.fit(X_bin, y_bin)
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_init_params():
    t = np.array([[-0.28, 0.95],
                  [-2.23, 2.4]])
    lr = TfSoftmaxRegression(epochs=50,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)

    lr.fit(X_bin, y_bin)
    lr.fit(X_bin, y_bin, init_params=False)
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_binary_logistic_regression_sgd():
    t = np.array([[0.35, 0.32],
                  [-7.14, 7.3]])
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=len(y_bin),
                             random_seed=1)

    lr.fit(X_bin, y_bin)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert (y_bin == lr.predict(X_bin)).all()


def test_multi_logistic_regression_gd_weights():
    t = np.array([[-0.94, -1.05, 2.73],
                  [-2.17, 2.01, 2.51]])
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)
    lr.fit(X, y)
    np.testing.assert_almost_equal(lr.w_, t, 2)


def test_multi_logistic_probas():
    lr = TfSoftmaxRegression(epochs=200,
                             eta=0.75,
                             minibatches=1,
                             random_seed=1)
    lr.fit(X, y)
    idx = [0, 50, 149]  # sample labels: 0, 1, 2
    y_pred = lr.predict_proba(X[idx])
    exp = np.array([[0.99, 0.01, 0.0],
                    [0.01, 0.89, 0.1],
                    [0.0, 0.02, 0.98]])
    np.testing.assert_almost_equal(y_pred, exp, 2)


def test_multi_logistic_regression_gd_acc():
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)
    lr.fit(X, y)
    assert (y == lr.predict(X)).all()


def test_score_function():
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)
    lr.fit(X, y)
    acc = lr.score(X, y)
    assert acc == 1.0, acc


def test_train_acc():
    lr = TfSoftmaxRegression(epochs=3,
                             eta=0.5,
                             minibatches=1,
                             random_seed=1)
    lr.fit(X, y)
    exp = [0.47, 0.65, 0.67]
    np.testing.assert_almost_equal(exp, lr.train_acc_, decimal=2)


@raises(AttributeError)
def test_fail_minibatches():
    lr = TfSoftmaxRegression(epochs=100,
                             eta=0.5,
                             minibatches=13,
                             random_seed=1)
    lr.fit(X, y)
    assert((y == lr.predict(X)).all())
