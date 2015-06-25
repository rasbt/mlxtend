from mlxtend.classifier import NeuralNetMLP
from mlxtend.data import iris_data
import numpy as np


## Iris Data
X, y = iris_data()

# standardize
X_std = np.copy(X)
for i in range(4):
    X_std[:,i] = (X[:,i] - X[:,i].mean()) / X[:,i].std()

def test_gradient_descent():

    nn = NeuralNetMLP(n_output=3, 
             n_features=X.shape[1], 
             n_hidden=10, 
             l2=0.0, 
             l1=0.0, 
             epochs=80, 
             eta=0.01, 
             minibatches=1, 
             shuffle=True,
             random_state=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert(round(acc, 2) == 0.97)


def test_minibatch():

    nn = NeuralNetMLP(n_output=3, 
             n_features=X.shape[1], 
             n_hidden=10, 
             l2=0.0, 
             l1=0.0, 
             epochs=80, 
             eta=0.01, 
             minibatches=10, 
             shuffle=True,
             random_state=1)

    nn.fit(X_std, y)
    y_pred = nn.predict(X_std)
    acc = np.sum(y == y_pred, axis=0) / float(X_std.shape[0])
    assert(round(acc, 2) == 0.97)
    
    
def test_binary():
    X0 = X_std[0:100] # class 0 and class 1
    y0 = y[0:100] # class 0 and class 1

    nn = NeuralNetMLP(n_output=3, 
             n_features=X0.shape[1], 
             n_hidden=10, 
             l2=0.0, 
             l1=0.1, 
             epochs=80, 
             eta=0.01, 
             minibatches=10, 
             shuffle=True,
             random_state=1)
    nn.fit(X0, y0)
    y_pred = nn.predict(X0)
    acc = np.sum(y0 == y_pred, axis=0) / float(X0.shape[0])
    assert(round(acc, 2) == 1.0)