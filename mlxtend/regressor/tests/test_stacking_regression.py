# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.regressor import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import raises
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.exceptions import NotFittedError
from mlxtend.utils import assert_raises


# Generating a sample dataset
np.random.seed(1)
X1 = np.sort(5 * np.random.rand(40, 1), axis=0)
X2 = np.sort(5 * np.random.rand(40, 2), axis=0)
y = np.sin(X1).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))
y2 = np.sin(X2)


def test_different_models():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge],
                               meta_regressor=svr_rbf)
    stregr.fit(X1, y).predict(X1)
    mse = 0.21
    got = np.mean((stregr.predict(X1) - y) ** 2)
    assert round(got, 2) == mse


def test_multivariate():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge],
                               meta_regressor=svr_rbf)
    stregr.fit(X2, y).predict(X2)
    mse = 0.22
    got = np.mean((stregr.predict(X2) - y) ** 2)
    assert round(got, 2) == mse


def test_multivariate_class():
    lr = LinearRegression()
    ridge = Ridge(random_state=1)
    meta = LinearRegression(normalize=True)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=meta)
    stregr.fit(X2, y2).predict(X2)
    mse = 0.12
    got = np.mean((stregr.predict(X2) - y2) ** 2.)
    # there seems to be an issue with the following test on Windows
    # sometimes via Appveyor
    assert round(got, 2) == mse, got


def test_gridsearch():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge],
                               meta_regressor=svr_rbf)

    params = {'ridge__alpha': [0.01, 1.0],
              'svr__C': [0.01, 1.0],
              'meta-svr__C': [0.01, 1.0]}

    grid = GridSearchCV(estimator=stregr,
                        param_grid=params,
                        cv=5,
                        refit=True,
                        verbose=0)
    grid = grid.fit(X1, y)
    best = 0.1
    got = round(grid.best_score_, 2)
    assert best == got


def test_gridsearch_numerate_regr():
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[svr_lin, ridge, ridge],
                               meta_regressor=svr_rbf)

    params = {'ridge-1__alpha': [0.01, 1.0],
              'ridge-2__alpha': [0.01, 1.0],
              'svr__C': [0.01, 1.0],
              'meta-svr__C': [0.01, 1.0]}

    grid = GridSearchCV(estimator=stregr,
                        param_grid=params,
                        cv=5,
                        refit=True,
                        verbose=0)
    grid = grid.fit(X1, y)
    best = 0.1
    got = round(grid.best_score_, 2)
    assert best == got


def test_get_coeff():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[svr_lin, lr],
                               meta_regressor=ridge)
    stregr.fit(X1, y)
    got = stregr.coef_
    expect = np.array([0.4874216, 0.45518317])
    assert_almost_equal(got, expect)


def test_get_intercept():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[svr_lin, lr],
                               meta_regressor=ridge)
    stregr.fit(X1, y)
    got = stregr.intercept_
    expect = 0.02
    assert round(got, 2) == expect


# ValueError was changed to AttributeError in sklearn >= 0.19
@raises(AttributeError, ValueError)
def test_get_coeff_fail():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[ridge, lr],
                               meta_regressor=svr_rbf)
    stregr = stregr.fit(X1, y)
    r = stregr.coef_
    assert r


def test_get_params():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[ridge, lr],
                               meta_regressor=svr_rbf)

    got = sorted(list({s.split('__')[0] for s in stregr.get_params().keys()}))
    expect = ['linearregression',
              'meta-svr',
              'meta_regressor',
              'regressors',
              'ridge',
              'store_train_meta_features',
              'verbose']
    assert got == expect, got


def test_regressor_gridsearch():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr],
                               meta_regressor=svr_rbf)

    params = {'regressors': [[lr], [lr, ridge]]}

    grid = GridSearchCV(estimator=stregr,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X1, y)

    assert len(grid.best_params_['regressors']) == 2


def test_predict_meta_features():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=svr_rbf)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)
    stregr.fit(X_train, y_train)
    test_meta_features = stregr.predict(X_test)
    assert test_meta_features.shape[0] == X_test.shape[0]


def test_train_meta_features_():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=svr_rbf,
                               store_train_meta_features=True)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)
    stregr.fit(X_train, y_train)
    train_meta_features = stregr.train_meta_features_
    assert train_meta_features.shape[0] == X_train.shape[0]


def test_not_fitted_predict():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=svr_rbf,
                               store_train_meta_features=True)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)

    expect = ("Estimator not fitted, "
              "call `fit` before exploiting the model.")

    assert_raises(NotFittedError,
                  expect,
                  stregr.predict,
                  X_train)


def test_not_fitted_predict_meta_features():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=svr_rbf,
                               store_train_meta_features=True)
    X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)

    expect = ("Estimator not fitted, "
              "call `fit` before exploiting the model.")

    assert_raises(NotFittedError,
                  expect,
                  stregr.predict_meta_features,
                  X_train)
