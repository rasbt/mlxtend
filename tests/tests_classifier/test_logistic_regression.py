from mlxtend.classifier import LogisticRegression
from mlxtend.data import iris_data
import numpy as np


X, y = iris_data()
X = X[:, [0, 3]] # sepal length and petal width
X = X[0:100] # class 0 and class 1
y = y[0:100] # class 0 and class 1

# standardize
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


def test_logistic_regression_gd():

    t = np.array([ 0.52, 1.20, 4.40])
    lr = LogisticRegression(epochs=100, eta=0.01, learning='gd')

    lr.fit(X, y) # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y == lr.predict(X)).all())



def test_logistic_regression_sgd():

    t = np.array([ 0.51, 1.18, 4.38])
    lr = LogisticRegression(epochs=100, eta=0.01, learning='sgd')

    lr.fit(X, y) # 0, 1 class
    np.testing.assert_almost_equal(lr.w_, t, 2)
    assert((y == lr.predict(X)).all())