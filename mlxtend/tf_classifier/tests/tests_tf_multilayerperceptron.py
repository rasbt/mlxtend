# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.tf_classifier import TfMultiLayerPerceptron as MLP
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


@raises(AttributeError)
def test_optimizer_init():
    MLP(optimizer='no-optimizer')


@raises(AttributeError)
def test_activations_init_typo():
    MLP(hidden_layers=[1, 2], activations=['logistic', 'invalid'])


@raises(AttributeError)
def test_activations_invalid_ele_1():
    MLP(hidden_layers=[1], activations=['logistic', 'logistic'])


@raises(AttributeError)
def test_activations_invalid_ele_2():
    MLP(hidden_layers=[10, 10], activations=['logistic'])


def test_mapping():
    mlp = MLP()
    w, b = mlp._layermapping(n_features=10,
                             n_classes=11,
                             hidden_layers=[8, 7, 6])

    expect_b = {1: [[8], 'n_hidden_1'],
                2: [[7], 'n_hidden_2'],
                3: [[6], 'n_hidden_3'],
                'out': [[11], 'n_classes']}

    expect_w = {1: [[10, 8], 'n_features, n_hidden_1'],
                2: [[8, 7], 'n_hidden_1, n_hidden_2'],
                3: [[7, 6], 'n_hidden_2, n_hidden_3'],
                'out': [[6, 11], 'n_hidden_3, n_classes']}

    assert expect_b == b, b
    assert expect_w == w, w


def test_binary_gd():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)

    mlp.fit(X_bin, y_bin)
    assert (y_bin == mlp.predict(X_bin)).all()


def test_binary_sgd():
    mlp = MLP(epochs=10,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=len(y_bin),
              random_seed=1)

    mlp.fit(X_bin, y_bin)
    assert (y_bin == mlp.predict(X_bin)).all()


def test_multiclass_probas():
    mlp = MLP(epochs=500,
              eta=0.5,
              hidden_layers=[10],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    idx = [0, 50, 149]  # sample labels: 0, 1, 2
    y_pred = mlp.predict_proba(X[idx])
    exp = np.array([[1.0, 0.0, 0.0],
                    [0.0, 0.9, 0.1],
                    [0.0, 0.1, 0.9]])
    np.testing.assert_almost_equal(y_pred, exp, 1)


def test_multiclass_gd_acc():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    assert (y == mlp.predict(X)).all()


def test_multiclass_gd_dropout():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1,
              dropout=0.05)
    mlp.fit(X, y)
    acc = round(mlp.score(X, y), 2)
    assert acc == 0.67, acc


def test_score_function():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


def test_score_function_momentum():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='momentum',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


def test_score_function_adam():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='adam',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


def test_score_function_ftrl():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='ftrl',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


def test_score_function_adagrad():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='adagrad',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)
    mlp.fit(X, y)
    acc = mlp.score(X, y)
    assert acc == 1.0, acc


@raises(AttributeError)
def test_fail_minibatches():
    mlp = MLP(epochs=100,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=13,
              random_seed=1)
    mlp.fit(X, y)
    assert (y == mlp.predict(X)).all()


def test_train_acc():
    mlp = MLP(epochs=3,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)

    mlp.fit(X, y)
    assert len(mlp.train_acc_) == 3


def test_valid_acc():
    mlp = MLP(epochs=3,
              eta=0.5,
              hidden_layers=[5],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              random_seed=1)

    mlp.fit(X, y, X_valid=X[:100], y_valid=y[:100])
    assert len(mlp.valid_acc_) == 3


def test_multiclass_gd_nolearningdecay():
    mlp = MLP(epochs=5,
              eta=0.5,
              hidden_layers=[15],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              decay=[0.0, 1.0],
              random_seed=1)
    mlp.fit(X, y)
    expect = [3.11, 2.12, 1.5, 1.17, 1.0]
    np.testing.assert_almost_equal(expect, mlp.cost_, decimal=2)


def test_multiclass_gd_learningdecay():
    mlp = MLP(epochs=5,
              eta=0.5,
              hidden_layers=[15],
              optimizer='gradientdescent',
              activations=['logistic'],
              minibatches=1,
              decay=[0.5, 1.0],
              random_seed=1)
    mlp.fit(X, y)
    expect = [3.11, 2.12, 1.79, 1.65, 1.59]
    np.testing.assert_almost_equal(expect, mlp.cost_, decimal=2)
