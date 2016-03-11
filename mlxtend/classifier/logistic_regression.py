# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Implementation of the logistic regression algorithm for classification.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .base import _BaseClassifier


class LogisticRegression(_BaseClassifier):
    """Logistic regression classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
    regularization : {None, 'l2'} (default: None)
        Type of regularization. No regularization if
        `regularization=None`.
    l2_lambda : float
        Regularization parameter for L2 regularization.
        No regularization if l2_lambda=0.0.
    minibatches : int (default: 1)
        Divide the training data into *k* minibatches
        for accelerated stochastic gradient descent learning.
        Gradient Descent Learning if `minibatches` = 1
        Stochastic Gradient Descent learning if `minibatches` = len(y)
        Minibatch learning if `minibatches` > 1
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.
    zero_init_weight : bool (default: False)
        If True, weights are initialized to zero instead of small random
        numbers following a standard normal distribution with mean=0 and
        stddev=1.
    print_progress : int (default: 0)
        Prints progress in fitting to stderr.
        0: No output
        1: Epochs elapsed and cost
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.
    cost_ : list
        List of floats with sum of squared error cost (sgd or gd) for every
        epoch.

    """
    def __init__(self, eta=0.01, epochs=50, regularization=None,
                 l2_lambda=0.0, minibatches=1,
                 random_seed=None, zero_init_weight=False,
                 print_progress=0):

        super(LogisticRegression, self).__init__(print_progress=print_progress)
        self.random_seed = random_seed
        self.eta = eta
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.minibatches = minibatches
        self.regularization = regularization
        self.zero_init_weight = zero_init_weight

        if self.regularization not in (None, 'l2'):
            raise AttributeError('regularization must be None or "l2"')

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
            (Re)initializes weights to small random floats if True.

        Returns
        -------
        self : object

        """
        self._check_arrays(X, y)

        if (np.unique(y) != np.array([0, 1])).all():
            raise ValueError('Supports only binary class labels 0 and 1')

        if init_weights:
            self.w_ = self._init_weights(shape=1 + X.shape[1],
                                         zero_init_weight=self.zero_init_weight,
                                         seed=self.random_seed)

        self.m_ = len(self.w_)
        self.cost_ = []

        # random seed for shuffling
        if self.random_seed:
            np.random.seed(self.random_seed)

        n_idx = list(range(y.shape[0]))
        self.init_time_ = time()
        for i in range(self.epochs):
            if self.minibatches > 1:
                n_idx = np.random.permutation(n_idx)

            minis = np.array_split(n_idx, self.minibatches)
            for idx in minis:
                y_val = self._activation(X[idx])
                errors = (y[idx] - y_val)
                neg_grad = X[idx].T.dot(errors)
                l2_reg = self.l2_lambda * self.w_[1:]
                self.w_[1:] += self.eta * (neg_grad - l2_reg)
                self.w_[0] += self.eta * errors.sum()

            cost = self._logit_cost(y, self._activation(X))
            self.cost_.append(cost)
            if self.print_progress:
                self._print_progress(epoch=i+1, cost=cost)
        return self

    def _predict(self, X):
        # equivalent to np.where(self._activation(X) >= 0.5, 1, 0)
        return np.where(self._net_input(X) >= 0.0, 1, 0)

    def _net_input(self, X):
        """Compute the linear net input."""
        return X.dot(self.w_[1:]) + self.w_[0]

    def _activation(self, X):
        """ Compute sigmoid activation."""
        z = self._net_input(X)
        return self._sigmoid(z)

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class 1 probability : float

        """
        return self._activation(X)

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        if self.l2_lambda:
            l2 = self.l2_lambda / 2.0 * np.sum(self.w_[1:]**2)
            logit += l2
        return logit

    def _sigmoid(self, z):
        """Compute the output of the logistic sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))
