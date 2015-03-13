from mlxtend.classifier import Adaline
from mlxtend.data import iris_data
import numpy as np


X, y = iris_data()
X = X[:, [0, 3]] # sepal length and petal width
X = X[0:100] # class 0 and class 1
y0 = y[0:100] # class 0 and class 1
y = np.where(y[0:100] == 0, -1, 1) # class -1 and class 1

# standardize
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

def test_gradient_descent():

    t1 = np.array([-5.20694599e-16, -7.86462587e-02, 1.02193041e+00])
    ada = Adaline(epochs=30, eta=0.01, learning='gd', random_state=1)
    ada.fit(X_std, y)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y == ada.predict(X_std)).all())


def test_stochastic_gradient_descent():

    t1 = np.array([0.02703854, -0.08923447, 1.01748196])
    ada = Adaline(epochs=30, eta=0.01, learning='sgd', random_state=1)
    ada.fit(X_std, y)
    np.testing.assert_almost_equal(ada.w_, t1, 2)
    assert((y == ada.predict(X_std)).all())

def test_invalid_class():

    ada = Adaline(epochs=40, eta=0.01, random_state=1)
    try:
        ada.fit(X, y0)  # 0, 1 class
        assert(1==2)
    except ValueError:
        pass
