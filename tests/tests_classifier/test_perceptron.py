from mlxtend.classifier import Perceptron
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

def test_standardized_data():

    t1 = np.array([0.177022, 0.40557785, 0.4977635])
    ppn = Perceptron(epochs=15, eta=0.01, random_state=1)

    ppn.fit(X_std, y)  # -1, 1 class
    np.testing.assert_almost_equal(ppn.w_, t1, 2)
    assert((y == ppn.predict(X_std)).all())

def test_nonstandardized_data():

    t1 = np.array([0.077022, -0.07367551, 0.46211437])
    ppn = Perceptron(epochs=40, eta=0.01, random_state=1)

    ppn.fit(X, y)  # -1, 1 class
    np.testing.assert_almost_equal(ppn.w_, t1, 2)
    assert((y == ppn.predict(X)).all())

def test_invalid_class():

    ppn = Perceptron(epochs=40, eta=0.01, random_state=1)
    try:
        ppn.fit(X, y0)  # 0, 1 class
        assert(1==2)
    except ValueError:
        pass
