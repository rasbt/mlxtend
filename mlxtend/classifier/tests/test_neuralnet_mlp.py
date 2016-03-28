# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier import NeuralNetMLP
from mlxtend.data import iris_data
import numpy as np


# Iris Data
X, y = iris_data()

# standardize
X_std = np.copy(X)
for i in range(4):
    X_std[:, i] = (X[:, i] - X[:, i].mean()) / X[:, i].std()


def test_gradient_checking():
    nn3 = NeuralNetMLP(n_output=len(np.unique(y)),
                       n_features=X_std.shape[1],
                       n_hidden=25,
                       l2=0.0,
                       l1=0.0,
                       epochs=1,
                       eta=0.01,
                       alpha=0.0,
                       decrease_const=0.0,
                       minibatches=1,
                       shuffle_init=False,
                       shuffle_epoch=False,
                       random_seed=1)

    for epoch in range(10):
        eucldist = nn3._gradient_checking(X=X_std, y=y)
        assert eucldist < 1e-07, 'Gradient difference is %s' % eucldist


def test_gradient_descent():

    nn = NeuralNetMLP(n_output=3,
                      n_features=X.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=100,
                      eta=0.1,
                      minibatches=1,
                      shuffle_init=False,
                      shuffle_epoch=False,
                      random_seed=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert round(acc, 2) == 0.99, "Acc: %s" % acc


def test_score_function():

    nn = NeuralNetMLP(n_output=3,
                      n_features=X.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=100,
                      eta=0.1,
                      minibatches=1,
                      shuffle_init=False,
                      shuffle_epoch=False,
                      random_seed=1)

    nn.fit(X_std, y)
    acc = nn.score(X_std, y)
    assert round(acc, 2) == 0.99, "Acc: %s" % acc


def test_shuffle():

    nn = NeuralNetMLP(n_output=3,
                      n_features=X.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=100,
                      eta=0.1,
                      minibatches=1,
                      shuffle_init=True,
                      shuffle_epoch=False,
                      random_seed=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert round(acc, 2) == 0.99, "Acc: %s" % acc

    nn = NeuralNetMLP(n_output=3,
                      n_features=X.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=100,
                      eta=0.1,
                      minibatches=1,
                      shuffle_init=True,
                      shuffle_epoch=True,
                      random_seed=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert round(acc, 2) == 0.99, "Acc: %s" % acc


def test_minibatch():
    nn = NeuralNetMLP(n_output=3,
                      n_features=X.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=30,
                      alpha=2.0,
                      eta=0.05,
                      minibatches=10,
                      shuffle_init=True,
                      shuffle_epoch=False,
                      random_seed=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert round(acc, 2) == 0.99, "Acc: %s" % acc


def test_binary():
    X0 = X_std[0:100]  # class 0 and class 1
    y0 = y[0:100]  # class 0 and class 1

    nn = NeuralNetMLP(n_output=2,
                      n_features=X0.shape[1],
                      n_hidden=10,
                      l2=0.0,
                      l1=0.0,
                      epochs=100,
                      eta=0.1,
                      minibatches=10,
                      shuffle_init=True,
                      shuffle_epoch=True,
                      random_seed=1)
    nn.fit(X0, y0)
    y_pred = nn.predict(X0)
    acc = np.sum(y0 == y_pred, axis=0) / float(X0.shape[0])
    assert round(acc, 2) == 1.0, "Acc: %s" % acc
