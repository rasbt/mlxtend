# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlxtend.classifier import OneRClassifier
from mlxtend.data import iris_data
from mlxtend.evaluate import create_counterfactual
from mlxtend.utils import assert_raises


def test__medium_lambda():
    X, y = iris_data()
    clf = LogisticRegression()
    clf.fit(X, y)

    x_ref = X[15]

    res = create_counterfactual(
        x_reference=x_ref,
        y_desired=2,
        model=clf,
        X_dataset=X,
        y_desired_proba=1.0,
        lammbda=1,
        random_seed=123,
    )

    assert np.argmax(clf.predict_proba(x_ref.reshape(1, -1))) == 0
    assert np.argmax(clf.predict_proba(res.reshape(1, -1))) == 2
    assert (
        round((clf.predict_proba(0.65 >= res.reshape(1, -1))).flatten()[-1], 2) <= 0.69
    )


def test__small_lambda():
    X, y = iris_data()
    clf = LogisticRegression()
    clf.fit(X, y)

    x_ref = X[15]

    res = create_counterfactual(
        x_reference=x_ref,
        y_desired=2,
        model=clf,
        X_dataset=X,
        y_desired_proba=1.0,
        lammbda=0.0001,
        random_seed=123,
    )

    assert np.argmax(clf.predict_proba(x_ref.reshape(1, -1))) == 0
    assert np.argmax(clf.predict_proba(res.reshape(1, -1))) == 0
    assert round((clf.predict_proba(res.reshape(1, -1))).flatten()[-1], 2) == 0.0


def test__large_lambda():
    X, y = iris_data()
    clf = LogisticRegression()
    clf.fit(X, y)

    x_ref = X[15]

    res = create_counterfactual(
        x_reference=x_ref,
        y_desired=2,
        model=clf,
        X_dataset=X,
        y_desired_proba=1.0,
        lammbda=100,
        random_seed=123,
    )

    assert np.argmax(clf.predict_proba(x_ref.reshape(1, -1))) == 0
    assert np.argmax(clf.predict_proba(res.reshape(1, -1))) == 2
    assert round((clf.predict_proba(res.reshape(1, -1))).flatten()[-1], 2) >= 0.96


def test__clf_with_no_proba_fail():
    X, y = iris_data()
    clf = OneRClassifier()
    clf.fit(X, y)

    x_ref = X[15]

    s = (
        "Your `model` does not support "
        "`predict_proba`. Set `y_desired_proba` "
        " to `None` to use `predict`instead."
    )

    assert_raises(
        AttributeError, s, create_counterfactual, x_ref, 2, clf, X, 1.0, 100, 123
    )


def test__clf_with_no_proba_pass():
    X, y = iris_data()
    clf = OneRClassifier()
    clf.fit(X, y)

    x_ref = X[15]

    res = create_counterfactual(
        x_reference=x_ref,
        y_desired=2,
        model=clf,
        X_dataset=X,
        y_desired_proba=None,
        lammbda=100,
        random_seed=123,
    )

    assert clf.predict(x_ref.reshape(1, -1)) == 0
    assert clf.predict(res.reshape(1, -1)) == 2
