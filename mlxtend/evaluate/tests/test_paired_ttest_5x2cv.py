# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_5x2cv


def test_classifier_defaults():
    X, y = iris_data()
    clf1 = LogisticRegression(random_state=1, multi_class="ovr", solver="liblinear")
    clf2 = DecisionTreeClassifier(random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score1, 2) == 0.97
    assert round(score2, 2) == 0.95

    t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf2, X=X, y=y, random_seed=1)

    assert round(t, 3) == -1.539, t
    assert round(p, 3) == 0.184, p

    # change maxdepth of decision tree classifier

    clf2 = DecisionTreeClassifier(max_depth=1, random_state=1)

    score3 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score3, 2) == 0.63

    t, p = paired_ttest_5x2cv(estimator1=clf1, estimator2=clf2, X=X, y=y, random_seed=1)

    assert round(t, 3) == 5.386, t
    assert round(p, 3) == 0.003, p


def test_scoring():
    X, y = iris_data()
    clf1 = LogisticRegression(random_state=1, solver="liblinear", multi_class="ovr")
    clf2 = DecisionTreeClassifier(random_state=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )

    score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
    score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

    assert round(score1, 2) == 0.97
    assert round(score2, 2) == 0.95

    t, p = paired_ttest_5x2cv(
        estimator1=clf1, estimator2=clf2, X=X, y=y, scoring="accuracy", random_seed=1
    )

    assert round(t, 3) == -1.539, t
    assert round(p, 3) == 0.184, p

    t, p = paired_ttest_5x2cv(
        estimator1=clf1, estimator2=clf2, X=X, y=y, scoring="f1_macro", random_seed=1
    )

    if Version(sklearn_version) < Version("0.20"):
        assert round(t, 3) == -1.510, t
        assert round(p, 3) == 0.191, p
    else:
        assert round(t, 3) == -1.506, t
        assert round(p, 3) == 0.192, p


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

    t, p = paired_ttest_5x2cv(estimator1=reg1, estimator2=reg2, X=X, y=y, random_seed=1)

    assert round(t, 3) == -0.599, t
    assert round(p, 3) == 0.575, p
