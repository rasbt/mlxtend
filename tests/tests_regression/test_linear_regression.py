from mlxtend.regression import LinearRegression
from mlxtend.data import iris_data
import numpy as np
from numpy.testing import assert_almost_equal

def test_univariate_normal_equation():
    X = np.array([ 1, 2, 3, 4, 5])[:, np.newaxis]
    y = np.array([ 1, 2, 3, 4, 5])
    ne_lr = LinearRegression(solver='normal_equation')
    ne_lr.fit(X, y)
    assert_almost_equal(ne_lr.w_, np.array([0.00, 1.00]), decimal=7)
    
def test_univariate_gradient_descent():
    X = np.array([ 1, 2, 3, 4, 5])[:, np.newaxis]
    y = np.array([ 1, 2, 3, 4, 5])
    X_std = (X - np.mean(X)) / X.std()
    y_std = (y - np.mean(y)) / y.std()
    
    gd_lr = LinearRegression(solver='gd', eta=0.1, epochs=50, random_seed=0)
    gd_lr.fit(X_std, y_std)
    assert_almost_equal(gd_lr.w_, np.array([0.00, 1.00]), decimal=2)

def test_univariate_stochastic_gradient_descent():
    X = np.array([ 1, 2, 3, 4, 5])[:, np.newaxis]
    y = np.array([ 1, 2, 3, 4, 5])
    X_std = (X - np.mean(X)) / X.std()
    y_std = (y - np.mean(y)) / y.std()
    
    sgd_lr = LinearRegression(solver='sgd', eta=0.1, epochs=10, random_seed=0)
    sgd_lr.fit(X_std, y_std)
    assert_almost_equal(sgd_lr.w_, np.array([0.00, 1.00]), decimal=2)


def test_multivariate_normal_equation():
    X = np.array([[1, 2], [2, 3], [4, 5], [6, 7], [7, 8]])
    y = np.array([ 1, 2, 3, 4, 5])
    X_std = (X - np.mean(X)) / X.std()
    y_std = (y - np.mean(y)) / y.std()
    ne_lr = LinearRegression(solver='normal_equation')
    ne_lr.fit(X_std, y_std)
    assert_almost_equal(ne_lr.predict(X_std), np.array(
            [ -1.3054, -0.87029, 0.0000, 0.8703, 1.3054]), 
            decimal=4)
    
def test_multivariate_gradient_descent():
    X = np.array([[1, 2], [2, 3], [4, 5], [6, 7], [7, 8]])
    y = np.array([ 1, 2, 3, 4, 5])
    X_std = (X - np.mean(X)) / X.std()
    y_std = (y - np.mean(y)) / y.std()

    gd_lr = LinearRegression(solver='gd', eta=0.01, epochs=100, random_seed=0)
    gd_lr.fit(X_std, y_std)
    assert_almost_equal(gd_lr.predict(X_std), np.array(
            [ -1.3054, -0.87029, 0.0000, 0.8703, 1.3054]), 
            decimal=2)

def test_multivariate_stochastic_gradient_descent():
    X = np.array([[1, 2], [2, 3], [4, 5], [6, 7], [7, 8]])
    y = np.array([ 1, 2, 3, 4, 5])
    X_std = (X - np.mean(X)) / X.std()
    y_std = (y - np.mean(y)) / y.std()
    
    sgd_lr = LinearRegression(solver='sgd', eta=0.001, epochs=1000, random_seed=0)
    sgd_lr.fit(X_std, y_std)
    assert_almost_equal(sgd_lr.predict(X_std), np.array(
            [ -1.3054, -0.87029, 0.0000, 0.8703, 1.3054]), 
            decimal=2)