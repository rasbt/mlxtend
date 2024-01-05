# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_resampled
from mlxtend.utils import assert_raises


def test_train_size():
    X, y = iris_data()
    clf1 = LogisticRegression(solver="liblinear", multi_class="ovr")
    clf2 = DecisionTreeClassifier()

    expected_err_msg = (
        "train_size must be of type int or float. " "Got <class 'NoneType'>."
    )

    if sys.version_info < (3, 0):
        expected_err_msg = expected_err_msg.replace("<class", "<type")

    assert_raises(
        ValueError,
        expected_err_msg,
        paired_ttest_resampled,
        clf1,
        clf2,
        X,
        y,
        test_size=None,
    )


def test_classifier_defaults():
    X, y = iris_data()
    clf1 = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=1)
    clf2 = DecisionTreeClassifier(random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score1, 2) == 0.97
    assert round(score2, 2) == 0.95

    t, p = paired_ttest_resampled(
        estimator1=clf1, estimator2=clf2, X=X, y=y, random_seed=1
    )

    if Version(sklearn_version) < Version("0.20"):
        assert round(t, 3) == -1.809, t
        assert round(p, 3) == 0.081, p
    else:
        assert round(t, 3) == -1.702, t
        assert round(p, 3) == 0.10, p

    # change maxdepth of decision tree classifier

    clf2 = DecisionTreeClassifier(max_depth=1, random_state=1)

    score3 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score3, 2) == 0.63

    t, p = paired_ttest_resampled(
        estimator1=clf1, estimator2=clf2, X=X, y=y, random_seed=1
    )

    assert round(t, 3) == 39.214, t
    assert round(p, 3) == 0.000, p


def test_scoring():
    X, y = iris_data()
    clf1 = LogisticRegression(multi_class="ovr", solver="liblinear", random_state=1)
    clf2 = DecisionTreeClassifier(random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score1, 2) == 0.97
    assert round(score2, 2) == 0.95

    t, p = paired_ttest_resampled(
        estimator1=clf1, estimator2=clf2, X=X, y=y, scoring="accuracy", random_seed=1
    )

    if Version(sklearn_version) < Version("0.20"):
        assert round(t, 3) == -1.809, t
        assert round(p, 3) == 0.081, p
    else:
        assert round(t, 3) == -1.702, t
        assert round(p, 3) == 0.1, p

    t, p = paired_ttest_resampled(
        estimator1=clf1, estimator2=clf2, X=X, y=y, scoring="f1_macro", random_seed=1
    )

    if Version(sklearn_version) < Version("0.20"):
        assert round(t, 3) == -1.690, t
        assert round(p, 3) == 0.102, p
    else:
        assert round(t, 3) == -1.561, t
        assert round(p, 3) == 0.129, p


def test_regressor():
    X, y = boston_housing_data()
    reg1 = Lasso(random_state=1)
    reg2 = Ridge(random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    score1 = reg1.fit(X_train, y_train).score(X_test, y_test)
    score2 = reg2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score1, 2) == 0.66, score1
    assert round(score2, 2) == 0.68, score2

    t, p = paired_ttest_resampled(
        estimator1=reg1, estimator2=reg2, X=X, y=y, random_seed=1
    )

    assert round(t, 3) == -7.697, t
    assert round(p, 3) == 0.000, p
