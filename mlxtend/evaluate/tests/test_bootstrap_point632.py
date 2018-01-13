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

X, y = iris_data()


def test_defaults():
    lr = LogisticRegression()
    scores = bootstrap_point632_score(lr, X, y, random_seed=123)
    acc = np.mean(scores)
    assert len(scores == 200)
    assert np.round(acc, 2) == 0.95


def test_invalid_splits():
    msg = 'Number of splits must be greater than 1. Got -1.'
    lr = LogisticRegression()
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  -1)


def test_allowed_methods():
    msg = "The `method` must be in ('.632', '.632+'). Got 1."
    lr = LogisticRegression()
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  200,
                  1)

    msg = "The `method` must be in ('.632', '.632+'). Got test."
    lr = LogisticRegression()
    assert_raises(ValueError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  200,
                  'test')


    msg = "The .632+ method is not implemented, yet."
    lr = LogisticRegression()
    assert_raises(NotImplementedError,
                  msg,
                  bootstrap_point632_score,
                  lr,
                  X,
                  y,
                  200,
                  '.632+')


def test_scoring():
    lr = LogisticRegression()
    scores = bootstrap_point632_score(lr, X[:100], y[:100],
                                      scoring='f1',
                                      random_seed=123)
    f1 = np.mean(scores)
    assert len(scores == 200)
    assert np.round(f1, 2) == 1.0, f1
