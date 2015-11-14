from mlxtend.classifier import Perceptron
from mlxtend.data import iris_data
import numpy as np


## Iris Data
X, y = iris_data()
X = X[:, [0, 3]] # sepal length and petal width
X = X[0:100] # class 0 and class 1
y0 = y[0:100] # class 0 and class 1
y1 = np.where(y[0:100] == 0, -1, 1) # class -1 and class 1
y2 = np.where(y[0:100] == 0, -2, 1) # class -2 and class 1

# standardize
X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


def test_standardized_iris_data():

    t1 = np.array([0.18, 0.41, 0.50])
    ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)

    ppn.fit(X_std, y1)  # -1, 1 class
    assert((y1 == ppn.predict(X_std)).all())


def test_nonstandardized_iris_data():

    t1 = np.array([0.078, -0.074, 0.46])
    ppn = Perceptron(epochs=40, eta=0.01, random_seed=1)
    ppn.fit(X, y1)  # -1, 1 class
    assert((y1 == ppn.predict(X)).all())


def test_0_1_class_iris_data():

    t1 = np.array([0.26, -0.  ,  0.27])
    ppn = Perceptron(epochs=40, eta=0.01, random_seed=1)
    ppn.fit(X, y0)  # 0, 1 class
    assert((y0 == ppn.predict(X)).all())


def test_invalid_class():

    ppn = Perceptron(epochs=40, eta=0.01, random_seed=1)
    try:
        ppn.fit(X, y2)  # -2, 1 class
        assert(1==2)
    except ValueError:
        pass
        
