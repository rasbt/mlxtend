# Sebastian Raschka 2014-2017
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
    mse = 0.214
    got = np.mean((stregr.predict(X1) - y) ** 2)
    assert round(got, 3) == mse


def test_multivariate():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stregr = StackingRegressor(regressors=[svr_lin, lr, ridge],
                               meta_regressor=svr_rbf)
    stregr.fit(X2, y).predict(X2)
    mse = 0.218
    got = np.mean((stregr.predict(X2) - y) ** 2)
    assert round(got, 3) == mse


def test_multivariate_class():
    lr = LinearRegression()
    ridge = Ridge(random_state=1)
    meta = LinearRegression(normalize=True)
    stregr = StackingRegressor(regressors=[lr, ridge],
                               meta_regressor=meta)
    stregr.fit(X2, y2).predict(X2)
    mse = 0.122
    got = np.mean((stregr.predict(X2) - y2) ** 2)
    assert round(got, 3) == mse


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
    best = 0.096
    got = round(grid.best_score_, 3)
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
    best = 0.096
    got = round(grid.best_score_, 3)
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
    expect = 0.024
    assert round(got, 3) == expect


@raises(ValueError)
def test_get_coeff_fail():
    lr = LinearRegression()
    svr_rbf = SVR(kernel='rbf')
    ridge = Ridge(random_state=1)
    stregr = StackingRegressor(regressors=[ridge, lr],
                               meta_regressor=svr_rbf)
    stregr = stregr.fit(X1, y)
    r = stregr.coef_
    assert r
