# Out-of-fold stacking regressor tests
#
# Sebastian Raschka 2014-2017
#
# mlxtend Machine Learning Library Extensions
# Author: Eike Dehling <e.e.dehling@gmail.com>
#
# License: BSD 3 clause

from mlxtend.regressor import OutOfFoldStackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import numpy as np
from sklearn.model_selection import GridSearchCV


# Some test data
np.random.seed(1)
X1 = np.sort(5 * np.random.rand(40, 1), axis=0)
X2 = np.sort(5 * np.random.rand(40, 2), axis=0)
X3 = np.zeros((40, 3))
y = np.sin(X1).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))
y2 = np.zeros((40,))


def test_different_models():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stack = OutOfFoldStackingRegressor(regressors=[svr_lin, lr, ridge],
                                       meta_regressor=svr_rbf)
    stack.fit(X1, y).predict(X1)
    mse = 0.21
    got = np.mean((stack.predict(X1) - y) ** 2)
    assert round(got, 2) == mse


def test_multivariate():
    lr = LinearRegression()
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stack = OutOfFoldStackingRegressor(regressors=[svr_lin, lr, ridge],
                                       meta_regressor=svr_rbf)
    stack.fit(X2, y).predict(X2)
    mse = 0.20
    got = np.mean((stack.predict(X2) - y) ** 2)
    assert round(got, 2) == mse


def test_internals():
    lr = LinearRegression()
    regressors = [lr, lr, lr, lr, lr]
    n_folds = 10
    stack = OutOfFoldStackingRegressor(regressors=[lr, lr, lr, lr, lr],
                                       meta_regressor=lr,
                                       n_folds=n_folds)
    stack.fit(X3, y2)
    assert stack.predict(X3).mean() == y2.mean()
    assert stack.meta_regr_.intercept_ == 0.0
    assert stack.meta_regr_.coef_[0] == 0.0
    assert stack.meta_regr_.coef_[1] == 0.0
    assert stack.meta_regr_.coef_[2] == 0.0
    assert len(stack.regr_) == len(regressors)
    for i, _ in enumerate(regressors):
        assert len(stack.regr_[i]) == n_folds


def test_gridsearch_numerate_regr():
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')
    stack = OutOfFoldStackingRegressor(regressors=[svr_lin, ridge, ridge],
                                       meta_regressor=svr_rbf)

    params = {'ridge-1__alpha': [0.01, 1.0],
              'ridge-2__alpha': [0.01, 1.0],
              'svr__C': [0.01, 1.0],
              'meta-svr__C': [0.01, 1.0]}

    grid = GridSearchCV(estimator=stack,
                        param_grid=params,
                        cv=5,
                        refit=True,
                        verbose=0)
    grid = grid.fit(X1, y)
    best = 0.12
    got = round(grid.best_score_, 2)
    assert best == got
