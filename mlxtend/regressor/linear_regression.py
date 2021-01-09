# Sebastian Raschka 2014-2020
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
    method : string (default: 'direct')
        For gradient descent-based optimization, use `sgd` (see `minibatch`
        parameter for further options). Otherwise, if `direct` (default),
        the analytical method is used. For alternative, numerically more
        stable solutions, use either `qr` (QR decomopisition) or `svd`
        (Singular Value Decomposition).
    eta : float (default: 0.01)
        solver learning rate (between 0.0 and 1.0). Used with `method =`
        `'sgd'`. (See `methods` parameter for details)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
        Used with `method = 'sgd'`. (See `methods` parameter for details)
    minibatches : int (default: None)
        The number of minibatches for gradient-based optimization.
        If None: Direct method, QR, or SVD method (see `method` parameter
                 for details)
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent learning
        If 1 < minibatches < len(y): Minibatch learning
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights. Used in
        `method = 'sgd'`. (See `methods` parameter for details)
    print_progress : int (default: 0)
        Prints progress in fitting to stderr if `method = 'sgd'`.
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

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/

    """
    def __init__(self, method='direct', eta=0.01, epochs=50,
                 minibatches=None, random_seed=None,
                 print_progress=0):

        _BaseModel.__init__(self)
        _IterativeModel.__init__(self)
        _Regressor.__init__(self)
        self.eta = eta
        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False
        self.method = method

        if method != 'sgd' and minibatches is not None:
            raise ValueError(('Minibatches should be set to `None` '
                              'if `method` != `sgd`. Got method=`%s`.')
                             % (method))

        supported_methods = ('sgd', 'direct', 'svd', 'qr')
        if method not in supported_methods:
            raise ValueError('`method` must be in %s. Got %s.' % (
                             supported_methods, method))

    def _fit(self, X, y, init_params=True):

        if init_params:
            self.b_, self.w_ = self._init_params(
                weights_shape=(X.shape[1], 1),
                bias_shape=(1,),
                random_seed=self.random_seed)
            self.cost_ = []

        # Direct analytical method
        if self.method == 'direct':
            self.b_, self.w_ = self._normal_equation(X, y)
        # Gradient descent or stochastic gradient descent learning
        elif self.method == 'sgd':
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
        # Solve using QR decomposition
        elif self.method == 'qr':
            Xb = np.hstack((np.ones((X.shape[0], 1)), X))
            Q, R = np.linalg.qr(Xb)
            beta = np.dot(np.linalg.inv(R), np.dot(Q.T, y))
            self.b_ = np.array([beta[0]])
            self.w_ = beta[1:].reshape(X.shape[1], 1)
        # Solve using SVD
        elif self.method == 'svd':
            Xb = np.hstack((np.ones((X.shape[0], 1)), X))
            beta = np.dot(np.linalg.pinv(Xb), y)
            self.b_ = np.array([beta[0]])
            self.w_ = beta[1:].reshape(X.shape[1], 1)

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
