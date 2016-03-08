# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Implementation of the ADAptive LInear NEuron classification algorithm.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .base import _BaseClassifier


class Adaline(_BaseClassifier):
    """ADAptive LInear NEuron classifier.

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
        numbers following a standard normal distribution with mean=0 and
        stddev=1;
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
      Sum of squared errors after each epoch.

    """
    def __init__(self, eta=0.01, epochs=50,
                 minibatches=None, random_seed=None,
                 zero_init_weight=False, print_progress=0):

        super(Adaline, self).__init__(print_progress=print_progress)
        self.random_seed = random_seed
        self.eta = eta
        self.minibatches = minibatches
        self.epochs = epochs
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

        # check if {0, 1} or {-1, 1} class labels are used
        self.classes_ = np.unique(y)
        if not len(self.classes_) == 2 \
                or not self.classes_[0] in (-1, 0) \
                or not self.classes_[1] == 1:
            raise ValueError('Only supports binary class'
                             ' labels {0, 1} or {-1, 1}.')
        if self.classes_[0] == -1:
            self.thres_ = 0.0
        else:
            self.thres_ = 0.5

        if init_weights:
            self.w_ = self._init_weights(shape=1 + X.shape[1],
                                         zero_init_weight=self.zero_init_weight,
                                         seed=self.random_seed)

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

    def _sum_squared_error_cost(self, y, y_val):
        errors = (y - y_val)
        return (errors**2).sum() / 2.0

    def _normal_equation(self, X, y):
        """Solve linear regression analytically."""
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.zeros(X.shape[1])
        z = np.linalg.inv(np.dot(Xb.T, Xb))
        w = np.dot(z, np.dot(Xb.T, y))
        return w

    def net_input(self, X):
        """Compute the linear net input."""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute the linear activation from the net input."""
        return self.net_input(X)

    def _predict(self, X):
        return np.where(self.net_input(X) >= self.thres_,
                        self.classes_[1], self.classes_[0])
