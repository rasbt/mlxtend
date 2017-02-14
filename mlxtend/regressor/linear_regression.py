# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Base Regressor (Regressor Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _Regressor


class LinearRegression(_BaseModel, _IterativeModel, _Regressor):

    """ Ordinary least squares linear regression.

    Parameters
    ------------
    eta : float (default: 0.01)
        solver rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    minibatches : int (default: None)
        The number of minibatches for gradient-based optimization.
        If None: Normal Equations (closed-form solution)
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent learning
        If 1 < minibatches < len(y): Minibatch learning
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.
    print_progress : int (default: 0)
        Prints progress in fitting to stderr if not solver='normal equation'
        0: No output
        1: Epochs elapsed and cost
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion

    Attributes
    -----------
    w_ : 2d-array, shape={n_features, 1}
      Model weights after fitting.
    b_ : 1d-array, shape={1,}
      Bias unit after fitting.
    cost_ : list
        Sum of squared errors after each epoch;
        ignored if solver='normal equation'

    """
    def __init__(self, eta=0.01, epochs=50,
                 minibatches=None, random_seed=None,
                 print_progress=0):

        self.eta = eta
        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True):

        if init_params:
            self.b_, self.w_ = self._init_params(
                weights_shape=(X.shape[1], 1),
                bias_shape=(1,),
                random_seed=self.random_seed)
            self.cost_ = []

        if self.minibatches is None:
            self.b_, self.w_ = self._normal_equation(X, y)

        # Gradient descent or stochastic gradient descent learning
        else:
            self.init_time_ = time()
            rgen = np.random.RandomState(self.random_seed)
            for i in range(self.epochs):

                for idx in self._yield_minibatches_idx(
                        rgen=rgen,
                        n_batches=self.minibatches,
                        data_ary=y,
                        shuffle=True):

                    y_val = self._net_input(X[idx])
                    errors = (y[idx] - y_val)
                    self.w_ += (self.eta *
                                X[idx].T.dot(errors).reshape(self.w_.shape))
                    self.b_ += self.eta * errors.sum()

                cost = self._sum_squared_error_cost(y, self._net_input(X))
                self.cost_.append(cost)
                if self.print_progress:
                    self._print_progress(iteration=(i + 1),
                                         n_iter=self.epochs,
                                         cost=cost)

        return self

    def _normal_equation(self, X, y):
        """Solve linear regression analytically."""
        Xb = np.hstack((np.ones((X.shape[0], 1)), X))
        w = np.zeros(X.shape[1])
        z = np.linalg.inv(np.dot(Xb.T, Xb))
        params = np.dot(z, np.dot(Xb.T, y))
        b, w = np.array([params[0]]), params[1:].reshape(X.shape[1], 1)
        return b, w

    def _net_input(self, X):
        """Compute the linear net input."""
        return (np.dot(X, self.w_) + self.b_).flatten()

    def _predict(self, X):
        return self._net_input(X)

    def _sum_squared_error_cost(self, y, y_val):
        errors = (y - y_val)
        return (errors**2).sum() / 2.0
