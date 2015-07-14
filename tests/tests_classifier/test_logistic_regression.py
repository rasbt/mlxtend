from mlxtend.classifier import LogisticRegression
from mlxtend.data import iris_data
import numpy as np


X, y = iris_data()
X = X[:, [0, 3]]  # sepal length and petal width
X = X[0:100]  # class 0 and class 1
y = y[0:100]  # class 0 and class 1

# standardize
X[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()


def test_logistic_regression_gd():
    t = np.array([0.55, 1.29, 4.39])
    lr = LogisticRegression(epochs=100, eta=0.01, learning='gd', random_seed=0)

    lr.fit(X, y)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y == lr.predict(X)).all())


def test_logistic_regression_sgd():
    t = np.array([0.55, 1.26, 4.39])
    lr = LogisticRegression(epochs=100, eta=0.01,
                            learning='sgd', random_seed=0)

    lr.fit(X, y)  # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y == lr.predict(X)).all())


def test_l2_regularization_gd():
    lr = LogisticRegression(eta=0.01,
                            epochs=20,
                            learning='gd',
                            l2_lambda=1.0,
                            regularization='l2',
                            random_seed=0)
    lr.fit(X, y)
    y_pred = lr.predict(X)
    expect_weights = np.array([0.252, 1.186, 2.296])

    np.testing.assert_almost_equal(lr.w_, expect_weights, 3)
    acc = sum(y_pred == y) / len(y)
    assert(acc == 1.0)


def test_l2_regularization_sgd():
    lr = LogisticRegression(eta=0.01, epochs=20,
                            learning='sgd',
                            l2_lambda=1.0,
                            regularization='l2',
                            random_seed=0)
    lr.fit(X, y)
    y_pred = lr.predict(X)
    expect_weights = np.array([0.100,  0.232,  0.346])

    np.testing.assert_almost_equal(lr.w_, expect_weights, 3)
    acc = sum(y_pred == y) / len(y)
    assert(acc == 1.0)
