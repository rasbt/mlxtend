# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Implementation of Rosenblatt's perceptron algorithm for classification.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .base import _BaseClassifier


class Perceptron(_BaseClassifier):
    """Perceptron classifier.

    Parameters
    ------------
    eta : float (default: 0.1)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Number of passes over the training dataset.
    shuffle : bool (default: False)
         Shuffles training data every epoch if True to prevent circles.
    random_seed : int
        Random state for initializing random weights.
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
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.1, epochs=50, shuffle=False,
                 random_seed=None, zero_init_weight=False,
                 print_progress=0):
        super(Perceptron, self).__init__(print_progress=print_progress)
        self.random_seed = random_seed
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
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
        self

        """
        self._check_arrays(X, y)

        # check if {0, 1} or {-1, 1} class labels are used
        self.classes_ = np.unique(y)
        if not (np.all(np.array([0, 1]) == self.classes_) or
                np.all(np.array([-1, 1]) == self.classes_)):
            raise ValueError('Only supports binary'
                             ' class labels {0, 1} or {-1, 1}.')

        if init_weights:
            self.w_ = self._init_weights(shape=1 + X.shape[1],
                                         zero_init_weight=self.zero_init_weight,
                                         seed=self.random_seed)

        self.cost_ = []

        # learn weights
        self.init_time_ = time()
        for i in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self._predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            if self.print_progress:
                self._print_progress(epoch=i+1, cost=errors)
            self.cost_.append(errors)
        return self

    def _net_input(self, X):
        """ Net input function """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def _predict(self, X):
        return np.where(self._net_input(X) >= 0.0,
                        self.classes_[1], self.classes_[0])
