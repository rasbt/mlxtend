# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.classifier import MultiLayerPerceptron as MLP
from mlxtend.data import iris_data
import numpy as np
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


def test_multiclass_gd_acc():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[0], 2) == 0.55, mlp.cost_[0]
    assert round(mlp.cost_[-1], 2) == 0.02, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()


def test_predict_proba():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)

    pred = mlp.predict_proba(X[0, np.newaxis])
    exp = np.array([[0.56, 0.22, 0.22]])
    np.testing.assert_almost_equal(pred, exp, decimal=2)


def test_multiclass_sgd_acc():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=len(y),
              random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 3) == 0.024, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()


def test_multiclass_minibatch_acc():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=5,
              random_seed=1)
    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 3) == 0.025, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()


def test_num_hidden_layers():
    assert_raises(AttributeError,
                  'Currently, only 1 hidden layer is supported',
                  MLP, 20, 0.05, [25, 10])


def test_binary_gd():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=5,
              random_seed=1)

    mlp.fit(X_bin, y_bin)
    assert (y_bin == mlp.predict(X_bin)).all()


def test_score_function():
    mlp = MLP(epochs=20,
              eta=0.05,
              hidden_layers=[25],
              minibatches=5,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


def test_decay_function():
    mlp = MLP(epochs=20,
              eta=0.05,
              decrease_const=0.01,
              hidden_layers=[25],
              minibatches=5,
              random_seed=1)

    mlp.fit(X, y)
    assert mlp._decr_eta < mlp.eta
    acc = mlp.score(X, y)
    assert round(acc, 2) == 0.98, acc


def test_momentum_1():
    mlp = MLP(epochs=20,
              eta=0.05,
              momentum=0.1,
              hidden_layers=[25],
              minibatches=len(y),
              random_seed=1)

    mlp.fit(X, y)
    assert round(mlp.cost_[-1], 4) == 0.0059, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()


def test_retrain():
    mlp = MLP(epochs=10,
              eta=0.05,
              hidden_layers=[25],
              minibatches=len(y),
              random_seed=1)

    mlp.fit(X, y)
    cost_1 = mlp.cost_[-1]
    mlp.fit(X, y, init_params=False)

    assert round(cost_1, 3) == 0.058, cost_1
    assert round(mlp.cost_[-1], 3) == 0.023, mlp.cost_[-1]
    assert (y == mlp.predict(X)).all()
