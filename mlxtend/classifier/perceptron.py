# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Implementation of Rosenblatt's perceptron algorithm for classification.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _Classifier


class Perceptron(_BaseModel, _IterativeModel, _Classifier):

    """Perceptron classifier.

    Note that this implementation of the Perceptron expects binary class labels
    in {0, 1}.

    Parameters
    ------------
    eta : float (default: 0.1)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Number of passes over the training dataset.
        Prior to each epoch, the dataset is shuffled to prevent cycles.
    random_seed : int
        Random state for initializing random weights and shuffling.
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
        Number of misclassifications in every epoch.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/Perceptron/

    """
    def __init__(self, eta=0.1, epochs=50, random_seed=None,
                 print_progress=0):

        _BaseModel.__init__(self)
        _IterativeModel.__init__(self)
        _Classifier.__init__(self)

        self.eta = eta
        self.epochs = epochs
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True):
        self._check_target_array(y, allowed={(0, 1)})
        y_data = np.where(y == 0, -1., 1.)

        if init_params:
            self.b_, self.w_ = self._init_params(
                weights_shape=(X.shape[1], 1),
                bias_shape=(1,),
                random_seed=self.random_seed)
            self.cost_ = []

        self.init_time_ = time()
        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):
            errors = 0

            for idx in self._yield_minibatches_idx(
                    rgen=rgen,
                    n_batches=y_data.shape[0], data_ary=y_data, shuffle=True):

                update = self.eta * (y_data[idx] -
                                     self._to_classlabels(X[idx]))
                self.w_ += (update * X[idx]).reshape(self.w_.shape)
                self.b_ += update
                errors += int(update != 0.0)

            if self.print_progress:
                self._print_progress(iteration=i + 1,
                                     n_iter=self.epochs,
                                     cost=errors)
            self.cost_.append(errors)
        return self

    def _net_input(self, X):
        """ Net input function """
        return (np.dot(X, self.w_) + self.b_).flatten()

    def _to_classlabels(self, X):
        return np.where(self._net_input(X) < 0.0, -1., 1.)

    def _predict(self, X):
        return np.where(self._net_input(X) < 0.0, 0, 1)
