# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from packaging.version import Version
from scipy import sparse
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.regressor import StackingRegressor
from mlxtend.utils import assert_raises

# Generating a sample dataset
np.random.seed(1)
X1 = np.sort(5 * np.random.rand(40, 1), axis=0)
X2 = np.sort(5 * np.random.rand(40, 2), axis=0)
y = np.sin(X1).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))
y2 = np.sin(X2)
w = np.random.random(40)


def test_different_models():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)
    stregr.fit(X1, y).predict(X1)
    mse = 0.21
    got = np.mean((stregr.predict(X1) - y) ** 2)
    assert round(got, 2) == mse


def test_multivariate():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)
    stregr.fit(X2, y).predict(X2)
    mse = 0.22
    got = np.mean((stregr.predict(X2) - y) ** 2)
    assert round(got, 2) == mse


def test_multivariate_class():
    lr = LinearRegression()
    ridge = Ridge(random_state=1)
    meta = LinearRegression()
    stregr = StackingRegressor(
        regressors=[lr, ridge], meta_regressor=meta, multi_output=True
    )
    stregr.fit(X2, y2).predict(X2)
    mse = 0.12
    got = np.mean((stregr.predict(X2) - y2) ** 2.0)
    assert round(got, 2) == mse, got


def test_sample_weight():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)
    pred1 = stregr.fit(X1, y, sample_weight=w).predict(X1)
    mse = 0.22
    got = np.mean((stregr.predict(X1) - y) ** 2)
    assert round(got, 2) == mse
    # make sure that this is not equivalent to the model with no weight
    pred2 = stregr.fit(X1, y).predict(X1)
    maxdiff = np.max(np.abs(pred1 - pred2))
    assert maxdiff > 1e-3, "max diff is %.4f" % maxdiff


def test_weight_ones():
    # sample weight of ones should produce equivalent outcome as no weight
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)
    pred1 = stregr.fit(X1, y).predict(X1)
    pred2 = stregr.fit(X1, y, sample_weight=np.ones(40)).predict(X1)
    maxdiff = np.max(np.abs(pred1 - pred2))
    assert maxdiff < 1e-3, "max diff is %.4f" % maxdiff


def test_weight_unsupported_regressor():
    # including regressor that does not support
    # sample_weight should raise error
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    knn = KNeighborsRegressor()
    stregr = StackingRegressor(
        regressors=[svr_lin, lr, ridge, knn], meta_regressor=svr_rbf
    )

    with pytest.raises(TypeError):
        stregr.fit(X1, y, sample_weight=w).predict(X1)


def test_weight_unsupported_meta():
    # meta regressor with no support for
    # sample_weight should raise error
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    knn = KNeighborsRegressor()
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=knn)

    with pytest.raises(TypeError):
        stregr.fit(X1, y, sample_weight=w).predict(X1)


def test_weight_unsupported_with_no_weight():
    # pass no weight to regressors with no weight support
    # should not be a problem
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    knn = KNeighborsRegressor()
    stregr = StackingRegressor(
        regressors=[svr_lin, lr, ridge, knn], meta_regressor=svr_rbf
    )
    stregr.fit(X1, y).predict(X1)

    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=knn)
    stregr.fit(X1, y).predict(X1)


def test_gridsearch():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], meta_regressor=svr_rbf)

    params = {
        "ridge__alpha": [0.01, 1.0],
        "svr__C": [0.01, 1.0],
        "meta_regressor__C": [0.01, 1.0],
    }

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            estimator=stregr, param_grid=params, cv=5, iid=False, refit=True, verbose=0
        )
    else:
        grid = GridSearchCV(
            estimator=stregr, param_grid=params, cv=5, refit=True, verbose=0
        )
    grid = grid.fit(X1, y)
    best = 0.1
    got = round(grid.best_score_, 2)
    assert best == got


def test_gridsearch_numerate_regr():
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stregr = StackingRegressor(
        regressors=[svr_lin, ridge, ridge], meta_regressor=svr_rbf
    )

    params = {
        "ridge-1__alpha": [0.01, 1.0],
        "ridge-2__alpha": [0.01, 1.0],
        "svr__C": [0.01, 1.0],
        "meta_regressor__C": [0.01, 1.0],
    }

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            estimator=stregr, param_grid=params, cv=5, iid=False, refit=True, verbose=0
        )
    else:
        grid = GridSearchCV(
            estimator=stregr, param_grid=params, cv=5, refit=True, verbose=0
        )
    grid = grid.fit(X1, y)
    best = 0.1
    got = round(grid.best_score_, 2)
    assert best == got


def test_get_coeff():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[svr_lin, lr], meta_regressor=ridge)
    stregr.fit(X1, y)
    got = stregr.coef_
    expect = np.array([0.4874216, 0.45518317])
    assert_almost_equal(got, expect)


def test_get_intercept():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[svr_lin, lr], meta_regressor=ridge)
    stregr.fit(X1, y)
    got = stregr.intercept_
    expect = 0.02
    assert round(got, 2) == expect


# ValueError was changed to AttributeError in sklearn >= 0.19
def test_get_coeff_fail():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[ridge, lr], meta_regressor=svr_rbf)

    with pytest.raises(AttributeError):
        stregr = stregr.fit(X1, y)
        r = stregr.coef_
        assert r


def test_get_params():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[ridge, lr], meta_regressor=svr_rbf)

    got = sorted(list({s.split("__")[0] for s in stregr.get_params().keys()}))
    expect = [
        "linearregression",
        "meta_regressor",
        "multi_output",
        "refit",
        "regressors",
        "ridge",
        "store_train_meta_features",
        "use_features_in_secondary",
        "verbose",
    ]
    assert got == expect, got


def test_regressor_gridsearch():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr], meta_regressor=svr_rbf)

    params = {"regressors": [[lr], [lr, ridge]]}

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            estimator=stregr, param_grid=params, cv=5, iid=False, refit=True
        )
    else:
        grid = GridSearchCV(estimator=stregr, param_grid=params, cv=5, refit=True)
    grid.fit(X1, y)

    assert len(grid.best_params_["regressors"]) == 2


def test_predict_meta_features():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr, ridge], meta_regressor=svr_rbf)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)
    stregr.fit(X_train, y_train)
    test_meta_features = stregr.predict(X_test)
    assert test_meta_features.shape[0] == X_test.shape[0]


def test_train_meta_features_():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(
        regressors=[lr, ridge], meta_regressor=svr_rbf, store_train_meta_features=True
    )
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)
    stregr.fit(X_train, y_train)
    train_meta_features = stregr.train_meta_features_
    assert train_meta_features.shape[0] == X_train.shape[0]


def test_not_fitted_predict():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(
        regressors=[lr, ridge], meta_regressor=svr_rbf, store_train_meta_features=True
    )
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)

    expect = (
        "This StackingRegressor instance is not fitted yet. Call "
        "'fit' with appropriate arguments before using this method."
    )

    assert_raises(NotFittedError, expect, stregr.predict, X_train)

    assert_raises(NotFittedError, expect, stregr.predict_meta_features, X_train)


def test_clone():
    lr = LinearRegression()
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(
        regressors=[lr, ridge], meta_regressor=svr_rbf, store_train_meta_features=True
    )
    clone(stregr)


def test_features_in_secondary():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    rf = RandomForestRegressor(n_estimators=10, random_state=2)
    ridge = Ridge(random_state=0)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stack = StackingRegressor(
        regressors=[svr_lin, lr, ridge, rf],
        meta_regressor=svr_rbf,
        use_features_in_secondary=True,
    )

    stack.fit(X1, y).predict(X1)
    mse = 0.14
    got = np.mean((stack.predict(X1) - y) ** 2)
    print(got)
    assert round(got, 2) == mse

    stack = StackingRegressor(
        regressors=[svr_lin, lr, ridge, rf],
        meta_regressor=svr_rbf,
        use_features_in_secondary=False,
    )

    # dense
    stack.fit(X1, y).predict(X1)
    mse = 0.12
    got = np.mean((stack.predict(X1) - y) ** 2)
    print(got)
    assert round(got, 2) == mse


def test_predictions_from_sparse_matrix():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[svr_lin, lr], meta_regressor=ridge)

    # dense
    stregr.fit(X1, y)
    print(stregr.score(X1, y))
    assert round(stregr.score(X1, y), 2) == 0.61

    # sparse
    stregr.fit(sparse.csr_matrix(X1), y)
    print(stregr.score(X1, y))
    assert round(stregr.score(X1, y), 2) == 0.61


def test_sparse_matrix_inputs_and_features_in_secondary():
    lr = LinearRegression()
    svr_lin = SVR(kernel="linear", gamma="auto")
    rf = RandomForestRegressor(n_estimators=10, random_state=2)
    ridge = Ridge(random_state=0)
    svr_rbf = SVR(kernel="rbf", gamma="auto")
    stack = StackingRegressor(
        regressors=[svr_lin, lr, ridge, rf],
        meta_regressor=svr_rbf,
        use_features_in_secondary=True,
    )

    # dense
    stack.fit(X1, y).predict(X1)
    mse = 0.14
    got = np.mean((stack.predict(X1) - y) ** 2)
    assert round(got, 2) == mse

    # sparse
    stack.fit(sparse.csr_matrix(X1), y)
    mse = 0.14
    got = np.mean((stack.predict(sparse.csr_matrix(X1)) - y) ** 2)
    assert round(got, 2) == mse
