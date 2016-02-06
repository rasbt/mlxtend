# Neural Network - Multilayer Perceptron

Implementation of a multilayer perceptron, a feedforward artificial neural network.

> from mlxtend.classifier import NeuralNetMLP

# Overview

*Although the code is fully working and can be used for common classification tasks, this implementation is not geared towards efficiency but clarity – the original code was written for demonstration purposes.*

## Basic Architecture

![](./NeuralNetMLP_files/neuralnet_mlp_1.png)  

The neurons $x_0$ and $a_0$ represent the bias units ($x_0=1$, $a_0=1$). 

The $i$th superscript denotes the $i$th layer, and the *j*th subscripts stands for the index of the respective unit. For example, $a_{1}^{(2)}$ refers to the first activation unit **after** the bias unit (i.e., 2nd activation unit) in the 2nd layer (here: the hidden layer)

  \begin{align}
    \mathbf{a^{(2)}} &= \begin{bmatrix}
           a_{0}^{(2)} \\
           a_{1}^{(2)} \\
           \vdots \\
           a_{m}^{(2)}
         \end{bmatrix}.
  \end{align}

Each layer $(l)$ in a multi-layer perceptron, a directed graph, is fully connected to the next layer $(l+1)$. We write the weight coefficient that connects the $k$th unit in the $l$th layer to the $j$th unit in layer $l+1$ as $w^{(l)}_{j, k}$.

For example, the weight coefficient that connects the units

$a_0^{(2)} \rightarrow a_1^{(3)}$

would be written as $w_{1,0}^{(2)}$.

## Activation

In the current implementation, the activations of the hidden and output layers are computed via the logistic (sigmoid) function $\phi(z) = \frac{1}{1 + e^{-z}}.$

![](./NeuralNetMLP_files/logistic_function.png)

(For more details on the logistic function, please see [`classifier.LogisticRegression`](./LogisticRegression.md); a general overview of different activation function can be found [here](../general_concepts/activation-functions.md).)

### References

- D. R. G. H. R. Williams and G. Hinton. [Learning representations by back-propagating errors](http://lia.disi.unibo.it/Courses/SistInt/articoli/nnet1.pdf). Nature, pages 323–533, 1986.
- C. M. Bishop. [Neural networks for pattern recognition](https://books.google.de/books?hl=en&lr=&id=T0S0BgAAQBAJ&oi=fnd&pg=PP1&dq=Neural+networks+for+pattern+recognition&ots=jL6TqGbBld&sig=fiLrMg-RJx22cgQ7zd2CiwUqNqI&redir_esc=y#v=onepage&q=Neural%20networks%20for%20pattern%20recognition&f=false). Oxford University Press, 1995.
- T. Hastie, J. Friedman, and R. Tibshirani. [The Elements of Statistical Learning](http://statweb.stanford.edu/%7Etibs/ElemStatLearn/), Volume 2. Springer, 2009.

# Examples

## Example 1 - Classifying Iris Flowers

Load 2 features from Iris (petal length and petal width) for visualization purposes:


```python
from mlxtend.data import iris_data
X, y = iris_data()
X = X[:, [0, 2]]    

# standardize training data
X_std = (X - X.mean(axis=0)) / X.std(axis=0)
```

Train neural network for 3 output flower classes ('Setosa', 'Versicolor', 'Virginica'), regular gradient decent (`minibatches=1`), 30 hidden units, and no regularization.

### Gradient Descent

Setting the `minibatches` to `1` will result in gradient descent training; please see [Gradient Descent vs. Stochastic Gradient Descent](../general_concepts/gradient-optimization.md) for details.


```python
from mlxtend.classifier import NeuralNetMLP

import numpy as np
nn1 = NeuralNetMLP(n_output=len(np.unique(y)), 
                   n_features=X_std.shape[1], 
                   n_hidden=50, 
                   l2=0.00, 
                   l1=0.0, 
                   epochs=300, 
                   eta=0.01, 
                   alpha=0.0,
                   decrease_const=0.0,
                   minibatches=1, 
                   shuffle_init=False,
                   shuffle_epoch=False,
                   random_seed=1,
                   print_progress=3)

nn1 = nn1.fit(X_std, y)
```

    Epoch: 300/300, Elapsed: 0:00:00, ETA: 0:00:00


```python
y_pred = nn1.predict(X_std)
acc = np.sum(y == y_pred, axis=0) / X_std.shape[0]
print('Accuracy: %.2f%%' % (acc * 100))
```

    Accuracy: 96.67%



```python
import matplotlib.pyplot as plt
plt.plot(range(len(nn1.cost_)), nn1.cost_)
plt.ylim([0, 300])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
```


![png](NeuralNetMLP_files/NeuralNetMLP_26_0.png)



```python
from mlxtend.evaluate import plot_decision_regions
fig = plot_decision_regions(X=X_std, y=y, clf=nn1, legend=2)
```


![png](NeuralNetMLP_files/NeuralNetMLP_27_0.png)


### Stochastic Gradient Descent

Setting `minibatches` to `n_samples` will result in stochastic gradient descent training; please see [Gradient Descent vs. Stochastic Gradient Descent](../general_concepts/gradient-optimization.md) for details.


```python
from mlxtend.classifier import NeuralNetMLP

import numpy as np
nn2 = NeuralNetMLP(n_output=len(np.unique(y)), 
                   n_features=X_std.shape[1], 
                   n_hidden=50, 
                   l2=0.00, 
                   l1=0.0, 
                   epochs=30, 
                   eta=0.01, 
                   alpha=0.2,
                   decrease_const=0.0,
                   minibatches=X_std.shape[0], 
                   shuffle_init=True,
                   shuffle_epoch=True,
                   random_seed=1,
                   print_progress=3)

nn2 = nn2.fit(X_std, y)
```

    Epoch: 30/30, Elapsed: 0:00:01, ETA: 0:00:00


```python
batches = np.array_split(range(len(nn2.cost_)), nn2.epochs+1)
cost_ary = np.array(nn2.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)),
         cost_avgs,
         color='red')
plt.ylim([0, 2])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
```


![png](NeuralNetMLP_files/NeuralNetMLP_31_0.png)


Continue the training for 200 epochs...


```python
nn2.epochs = 200
nn2 = nn2.fit(X_std, y)
```

    Epoch: 200/200, Elapsed: 0:00:05, ETA: 0:00:00


```python
batches = np.array_split(range(len(nn2.cost_)), nn2.epochs+1)
cost_ary = np.array(nn2.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(30, len(cost_avgs)+30),
         cost_avgs,
         color='red')
plt.ylim([0, 2])
plt.xlim([30, 30+nn2.epochs])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
plt.show()
```


![png](NeuralNetMLP_files/NeuralNetMLP_34_0.png)



```python
y_pred = nn2.predict(X_std)
acc = np.sum(y == y_pred, axis=0) / X_std.shape[0]
print('Accuracy: %.2f%%' % (acc * 100))
```

    Accuracy: 96.67%



```python
fig = plot_decision_regions(X=X_std, y=y, clf=nn2, legend=2)
plt.show()
```


![png](NeuralNetMLP_files/NeuralNetMLP_36_0.png)


## Example 2 - Classifying Handwritten Digits from a 10% MNIST Subset

Load a **5000-sample subset** of the [MNIST dataset](http://rasbt.github.io/mlxtend/docs/data/mnist/) (please see [`data.load_mnist`](../data/load_mnist.md) if you want to download and read in the complete MNIST dataset).



```python
from mlxtend.data import mnist_data
from mlxtend.preprocessing import shuffle_arrays_unison

X, y = mnist_data()
X, y = shuffle_arrays_unison((X, y), random_seed=1)
X_train, y_train = X[:500], y[:500]
X_test, y_test = X[500:], y[500:]
```

Visualize a sample from the MNIST dataset to check if it was loaded correctly:


```python
import matplotlib.pyplot as plt

def plot_digit(X, y, idx):
    img = X[idx].reshape(28,28)
    plt.imshow(img, cmap='Greys',  interpolation='nearest')
    plt.title('true label: %d' % y[idx])
    plt.show()
    
plot_digit(X, y, 3500)    
```


![png](NeuralNetMLP_files/NeuralNetMLP_41_0.png)


Standardize pixel values:


```python
import numpy as np
from mlxtend.preprocessing import standardize

X_train_std, params = standardize(X_train, 
                                  columns=range(X_train.shape[1]), 
                                  return_params=True)

X_test_std = standardize(X_test,
                         columns=range(X_test.shape[1]),
                         params=params)
```

Initialize the neural network to recognize the 10 different digits (0-10) using 300 epochs and mini-batch learning.


```python
nn = NeuralNetMLP(n_output=10, 
                  n_features=X_train_std.shape[1],
                  n_hidden=50,
                  l2=0.5,
                  l1=0.0,
                  epochs=300,
                  eta=0.001,
                  minibatches=25,
                  alpha=0.001,
                  decrease_const=0.0,
                  random_seed=1,
                  print_progress=3)
```

Learn the features while printing the progress to get an idea about how long it may take.


```python
import matplotlib.pyplot as plt

nn.fit(X_train_std, y_train)

plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 500])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.show()
```

    Epoch: 300/300, Elapsed: 0:00:12, ETA: 0:00:00


![png](NeuralNetMLP_files/NeuralNetMLP_47_1.png)



```python
y_train_pred = nn.predict(X_train_std)
y_test_pred = nn.predict(X_test_std)

train_acc = np.sum(y_train == y_train_pred, axis=0) / X_train_std.shape[0]
test_acc = np.sum(y_test == y_test_pred, axis=0) / X_test_std.shape[0]

print('Train Accuracy: %.2f%%' % (train_acc * 100))
print('Test Accuracy: %.2f%%' % (test_acc * 100))
```

    Train Accuracy: 98.80%
    Test Accuracy: 84.00%


**Please note** that this neural network has been trained on only 10% of the MNIST data for technical demonstration purposes, hence, the lousy predictive performance.

# API


*NeuralNetMLP(n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, random_weights=[-1.0, 1.0], shuffle_init=True, shuffle_epoch=True, minibatches=1, random_seed=None, print_progress=0)*

Feedforward neural network / Multi-layer perceptron classifier.

**Parameters**

- `n_output` : int

    Number of output units, should be equal to the
    number of unique class labels.

- `n_features` : int

    Number of features (dimensions) in the target dataset.
    Should be equal to the number of columns in the X array.

- `n_hidden` : int (default: 30)

    Number of hidden units.

- `l1` : float (default: 0.0)

    Lambda value for L1-regularization.
    No regularization if l1=0.0 (default)

- `l2` : float (default: 0.0)

    Lambda value for L2-regularization.
    No regularization if l2=0.0 (default)

- `epochs` : int (default: 500)

    Number of passes over the training set.

- `eta` : float (default: 0.001)

    Learning rate.

- `alpha` : float (default: 0.0)

    Momentum constant. Factor multiplied with the
    gradient of the previous epoch t-1 to improve
    learning speed
    w(t) := w(t) - (grad(t) + alpha*grad(t-1))

- `decrease_const` : float (default: 0.0)

    Decrease constant. Shrinks the learning rate
    after each epoch via eta / (1 + epoch*decrease_const)

- `random_weights` : list (default: [-1.0, 1.0])

    Min and max values for initializing the random weights.
    Initializes weights to 0 if None or False.

- `shuffle_init` : bool (default: True)

    Shuffles (a copy of the) training data before training.

- `shuffle_epoch` : bool (default: True)

    Shuffles training data before every epoch if True to prevent circles.

- `minibatches` : int (default: 1)

    Divides training data into k minibatches for efficiency.
    Normal gradient descent learning if k=1 (default).

- `random_seed` : int (default: None)

    Set random seed for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
    0: No output
    1: Epochs elapsed
    2: 1 plus time elapsed
    3: 2 plus estimated time until completion

**Attributes**

- `cost_` : list

    Sum of squared errors after each epoch.

### Methods

<hr>

*fit(X, y)*

Learn weight coefficients from training data.

**Parameters**

- `X` : array, shape = [n_samples, n_features]

    Input layer with original features.

- `y` : array, shape = [n_samples]

    Target class labels.

**Returns:**

self

<hr>

*predict(X)*

Predict class labels

**Parameters**

- `X` : array, shape = [n_samples, n_features]

    Input layer with original features.

**Returns:**

- `y_pred` : array, shape = [n_samples]

    Predicted class labels.


