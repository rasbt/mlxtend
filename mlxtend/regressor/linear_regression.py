# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Regressor (Regressor Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sys import stderr
from time import time
from .base import _BaseRegressor

# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Class for fitting a linear regression model.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class LinearRegression(_BaseRegressor):
    """ Ordinary least squares linear regression.

    Parameters
    ------------
    eta : float (default: 0.01)
        solver rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
    minibatches : int (default: None)
        The number of minibatches for gradient-based optimization.
        If None: Normal Equations (closed-form solution)
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent learning
        If 1 < minibatches < len(y): Minibatch learning
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.
    zero_init_weight : bool (default: False)
        If True, weights are initialized to zero instead of small random
        numbers in the interval [-0.1, 0.1];
        ignored if solver='normal equation'
    print_progress : int (default: 0)
        Prints progress in fitting to stderr if not solver='normal equation'
        0: No output
        1: Epochs elapsed and cost
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        Sum of squared errors after each epoch;
        ignored if solver='normal equation'

    """
    def __init__(self, eta=0.01, epochs=50,
                 minibatches=None, random_seed=None,
                 zero_init_weight=False, print_progress=0):

        np.random.seed(random_seed)
        self.eta = eta
        self.epochs = epochs
        self.minibatches = minibatches
        self.print_progress = print_progress
        self.zero_init_weight = zero_init_weight

    def fit(self, X, y, init_weights=True):
        """Learn weight coefficients from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_weights : bool (default: True)
            Re-initializes weights prior to fitting. Set False to continue
            training with weights from a previous fitting.

        Returns
        -------
        self : object

        """
        self._check_arrays(X, y)

        # initialize weights
        if init_weights:
            self._init_weights(shape=1 + X.shape[1])

        self.cost_ = []

        if self.minibatches is None:
            self.w_ = self._normal_equation(X, y)

        # Gradient descent or stochastic gradient descent learning
        else:
            n_idx = list(range(y.shape[0]))
            self.init_time_ = time()
            for i in range(self.epochs):
                if self.minibatches > 1:
                    X, y = self._shuffle(X, y)

                minis = np.array_split(n_idx, self.minibatches)
                for idx in minis:
                    y_val = self.activation(X[idx])
                    errors = (y[idx] - y_val)
                    self.w_[1:] += self.eta * X[idx].T.dot(errors)
                    self.w_[0] += self.eta * errors.sum()

                cost = self._sum_squared_error_cost(y, self.activation(X))
                self.cost_.append(cost)
                if self.print_progress:
                    self._print_progress(epoch=i+1, cost=cost)

        return self

    def _init_weights(self, shape):
        """Initialize weight coefficients."""
        if self.zero_init_weight:
            self.w_ = np.zeros(shape)
        else:
            self.w_ = 0.2 * np.random.ranf(shape) - 0.5
        self.w_.astype('float64')

    def _normal_equation(self, X, y):
        """Solve linear regression analytically."""
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        z = np.linalg.inv(np.dot(Xb.T, Xb))
        w = np.dot(z, np.dot(Xb.T, y))
        return w

    def _shuffle(self, X, y):
        """Unison shuffling."""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X):
        """Compute the linear net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute the linear activation from the net input."""
        return self.net_input(X)

    def predict(self, X):
        """Predict class labels of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        float : Predicted target value.

        """
        return self.net_input(X)

    def _sum_squared_error_cost(self, y, y_val):
        errors = (y - y_val)
        return (errors**2).sum() / 2.0
