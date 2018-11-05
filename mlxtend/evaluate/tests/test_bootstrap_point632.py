# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.evaluate import bootstrap_point632_score
from mlxtend.utils import assert_raises
from mlxtend.data import iris_data
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X, y = iris_data()


def test_defaults():
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    scores = bootstrap_point632_score(lr, X, y, random_seed=123)
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.95, np.round(acc, 2)


def test_oob():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123, method='oob')
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.95, np.round(acc, 2)


def test_632():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123,
                                      method='.632')
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.97, np.round(acc, 2)

    tree2 = DecisionTreeClassifier(random_state=123, max_depth=1)
    scores = bootstrap_point632_score(tree2, X, y, random_seed=123,
                                      method='.632')
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.66, np.round(acc, 2)


def test_632plus():
    tree = DecisionTreeClassifier(random_state=123)
    scores = bootstrap_point632_score(tree, X, y, random_seed=123,
                                      method='.632+')
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.97, np.round(acc, 2)

    tree2 = DecisionTreeClassifier(random_state=123, max_depth=1)
    scores = bootstrap_point632_score(tree2, X, y, random_seed=123,
                                      method='.632+')
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.65, np.round(acc, 2)


def test_custom_accuracy():

    def accuracy2(targets, predictions):
        return sum([i == j for i, j in
                    zip(targets, predictions)]) / len(targets)
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    scores = bootstrap_point632_score(lr, X, y,
                                      random_seed=123,
                                      scoring_func=accuracy2)
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.95, np.round(acc, 2)


def test_invalid_splits():
    msg = 'Number of splits must be greater than 1. Got -1.'
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  -1)


def test_allowed_methods():
    msg = "The `method` must be in ('.632', '.632+', 'oob'). Got 1."
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  200,
                  1)

    msg = "The `method` must be in ('.632', '.632+', 'oob'). Got test."
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  200,
                  'test')


def test_scoring():
    from sklearn.metrics import f1_score
    lr = LogisticRegression(solver='liblinear', multi_class='ovr')
    scores = bootstrap_point632_score(lr, X[:100], y[:100],
                                      scoring_func=f1_score,
                                      random_seed=123)
    f1 = np.mean(scores)
    assert len(scores == 200)
    assert np.round(f1, 2) == 1.0, f1
