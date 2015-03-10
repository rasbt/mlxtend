from mlxtend.classifier import Perceptron
from mlxtend.data import iris_data
import numpy as np


X, y = iris_data()
X = X[:, [0, 3]] # sepal length and petal width
X = X[0:100] # class 0 and class 1
y1 = y[0:100] # class 0 and class 1
y2 = np.asarray([50 * [-1] + 50 * [1]]).flatten()

# standardize
X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


def test_perceptron_rule():

    t1 = np.array([ 0.45, 0.069, 0.31])
    ppn = Perceptron(epochs=15, eta=0.01, learning='perceptron')

    ppn.fit(X, y1) # 0, 1 class
    np.testing.assert_almost_equal(ppn.w_, t1, 2)
    assert((y1 == ppn.predict(X)).all())

    t2 = np.array([ 0.0, 0.01, 0.04])
    ppn.fit(X, y2)  # -1, 1 class
    np.testing.assert_almost_equal(ppn.w_, t2, 2)
    assert((y2 == ppn.predict(X)).all())


def test_sgd():

    t1 = np.array([ 0.51, -0.03, 0.5])
    ppn = Perceptron(epochs=15, eta=0.01, learning='sgd')

    ppn.fit(X, y1) # 0, 1 class
    np.testing.assert_almost_equal(ppn.w_, t1, 2)
    assert((y1 == ppn.predict(X)).all())

    t2 = np.array([ 0.03, -0.07, 1.0])
    ppn.fit(X, y2)  # -1, 1 class
    np.testing.assert_almost_equal(ppn.w_, t2, 2)
    assert((y2 == ppn.predict(X)).all())


def test_gd():

    t1 = np.array([ 0.51, -0.03, 0.5])
    ppn = Perceptron(epochs=15, eta=0.01, learning='sgd')

    ppn.fit(X, y1) # 0, 1 class
    np.testing.assert_almost_equal(ppn.w_, t1, 2)
    assert((y1 == ppn.predict(X)).all())

    t2 = np.array([ 0.03, -0.07, 1.0])
    ppn.fit(X, y2)  # -1, 1 class
    np.testing.assert_almost_equal(ppn.w_, t2, 2)
    assert((y2 == ppn.predict(X)).all())