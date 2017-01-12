# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Implementation of the logistic regression algorithm for classification.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _Classifier


class LogisticRegression(_BaseModel, _IterativeModel, _Classifier):

    """Logistic regression classifier.

    Note that this implementation of Logistic Regression
    expects binary class labels in {0, 1}.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    l2_lambda : float
        Regularization parameter for L2 regularization.
        No regularization if l2_lambda=0.0.
    minibatches : int (default: 1)
        The number of minibatches for gradient-based optimization.
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent (SGD) online learning
        If 1 < minibatches < len(y): SGD Minibatch learning
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.
    print_progress : int (default: 0)
        Prints progress in fitting to stderr.
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
        List of floats with cross_entropy cost (sgd or gd) for every
        epoch.

    """
    def __init__(self, eta=0.01, epochs=50,
                 l2_lambda=0.0, minibatches=1,
                 random_seed=None,
                 print_progress=0):

        self.eta = eta
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True):

        self._check_target_array(y, allowed={(0, 1)})

        if init_params:
            self.b_, self.w_ = self._init_params(
                weights_shape=(X.shape[1], 1),
                bias_shape=(1,),
                random_seed=self.random_seed)
            self.cost_ = []

        self.init_time_ = time()
        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):

            for idx in self._yield_minibatches_idx(
                    rgen=rgen,
                    n_batches=self.minibatches,
                    data_ary=y,
                    shuffle=True):

                y_val = self._activation(X[idx])
                errors = (y[idx] - y_val)
                neg_grad = X[idx].T.dot(errors).reshape(self.w_.shape)
                l2_reg = self.l2_lambda * self.w_
                self.w_ += self.eta * (neg_grad - l2_reg)
                self.b_ += self.eta * errors.sum()

            cost = self._logit_cost(y, self._activation(X))
            self.cost_.append(cost)
            if self.print_progress:
                self._print_progress(iteration=(i + 1),
                                     n_iter=self.epochs,
                                     cost=cost)
        return self

    def _predict(self, X):
        # equivalent to np.where(self._activation(X) < 0.5, 0, 1)
        return np.where(self._net_input(X) < 0.0, 0, 1)

    def _net_input(self, X):
        """Compute the linear net input."""
        return (X.dot(self.w_) + self.b_).flatten()

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
