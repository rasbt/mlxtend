# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import bootstrap_point632_score
from mlxtend.utils import assert_raises

X, y = iris_data()


class FakeClassifier(BaseEstimator):
    def __init__(self):
        pass


def test_pandas_pass():
    tree = DecisionTreeClassifier(random_state=123)
    X_df = pd.DataFrame(X)
    y_ser = pd.Series(y)
    bootstrap_point632_score(tree, X_df, y_ser, random_seed=123, method="oob")
    bootstrap_point632_score(tree, X_df, y_ser, random_seed=123, method=".632")
    bootstrap_point632_score(tree, X_df, y_ser, random_seed=123, method=".632+")


def test_defaults():
    lr = LogisticRegression(solver="lbfgs")
    scores = bootstrap_point632_score(lr, X, y, random_seed=123)
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.96885, np.round(acc, 5)


def test_oob():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123, method="oob")
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.94667, np.round(acc, 5)


def test_632():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123, method=".632")
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.95914, np.round(acc, 5)

    tree2 = DecisionTreeClassifier(random_state=123, max_depth=1)
    scores = bootstrap_point632_score(tree2, X, y, random_seed=123, method=".632")
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.64355, np.round(acc, 5)


def test_632plus():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123, method=".632+")
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.95855, np.round(acc, 5)

    tree2 = DecisionTreeClassifier(random_state=123, max_depth=1)
    scores = bootstrap_point632_score(tree2, X, y, random_seed=123, method=".632+")
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.6409, np.round(acc, 5)


def test_632plus_overfitting_rate_is_clipped():
    # a fully grown tree memorizes random labels (training error 0), so the
    # out-of-bag error can be larger than the no-information error rate
    rng = np.random.RandomState(0)
    X_noise = rng.randn(40, 2)
    y_noise = rng.randint(0, 2, 40)
    tree = DecisionTreeClassifier(random_state=0)
    scores = bootstrap_point632_score(
        tree, X_noise, y_noise, n_splits=200, method=".632+", random_seed=1
    )
    assert scores.min() >= 0.0, scores.min()
    assert scores.max() <= 1.0, scores.max()


def test_632plus_no_overfitting_equals_632():
    # a constant model has a training error equal to the no-information
    # error rate, so R is 0 and .632+ reduces to .632
    dummy = DummyClassifier(strategy="most_frequent")
    scores_632 = bootstrap_point632_score(
        dummy, X, y, n_splits=20, method=".632", random_seed=1
    )
    scores_632plus = bootstrap_point632_score(
        dummy, X, y, n_splits=20, method=".632+", random_seed=1
    )
    np.testing.assert_allclose(scores_632plus, scores_632)


def test_custom_accuracy():
    def accuracy2(targets, predictions):
        return sum([i == j for i, j in zip(targets, predictions)]) / len(targets)

    lr = LogisticRegression(solver="lbfgs")
    scores = bootstrap_point632_score(lr, X, y, random_seed=123, scoring_func=accuracy2)
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 5) == 0.96885, np.round(acc, 5)


def test_invalid_splits():
    msg = "Number of splits must be greater than 1. Got -1."
    lr = LogisticRegression(solver="lbfgs")
    assert_raises(ValueError, msg, bootstrap_point632_score, lr, X, y, -1)


def test_allowed_methods():
    msg = "The `method` must be in ('.632', '.632+', 'oob'). Got 1."
    lr = LogisticRegression(solver="lbfgs")
    assert_raises(ValueError, msg, bootstrap_point632_score, lr, X, y, 200, 1)

    msg = "The `method` must be in ('.632', '.632+', 'oob'). Got test."
    lr = LogisticRegression(solver="lbfgs")
    assert_raises(ValueError, msg, bootstrap_point632_score, lr, X, y, 200, "test")


def test_scoring():
    from sklearn.metrics import f1_score

    lr = LogisticRegression(solver="lbfgs")
    scores = bootstrap_point632_score(
        lr, X[:100], y[:100], scoring_func=f1_score, random_seed=123
    )
    f1 = np.mean(scores)
    assert len(scores == 200)
    assert np.round(f1, 2) == 1.0, f1


def test_scoring_proba():
    from sklearn.metrics import roc_auc_score

    lr = LogisticRegression(solver="lbfgs")

    # test predict_proba
    scores = bootstrap_point632_score(
        lr,
        X[:100],
        y[:100],
        scoring_func=roc_auc_score,
        predict_proba=True,
        random_seed=123,
    )
    roc_auc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(roc_auc, 2) == 1.0, roc_auc

    with pytest.raises(RuntimeError):
        clf = FakeClassifier()
        scores = bootstrap_point632_score(
            clf,
            X[:100],
            y[:100],
            scoring_func=roc_auc_score,
            predict_proba=True,
            random_seed=123,
        )


if "TRAVIS" in os.environ or os.environ.get("TRAVIS") == "true":
    TRAVIS = True
else:
    TRAVIS = False

if "APPVEYOR" in os.environ or os.environ.get("APPVEYOR") == "true":
    APPVEYOR = True
else:
    APPVEYOR = False

GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS_CI", "false").lower() == "true"


@pytest.mark.skipif(
    TRAVIS or APPVEYOR or GITHUB_ACTIONS, reason="TensorFlow dependency"
)
def test_keras_fitparams():
    import tensorflow as tf

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(32, activation=tf.nn.relu), tf.keras.layers.Dense(1)]
    )

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    model.fit(X, y, epochs=5)
