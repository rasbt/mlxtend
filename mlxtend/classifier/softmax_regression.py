# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Implementation of the mulitnomial logistic regression algorithm for
# classification.

# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from time import time
from .base import _BaseClassifier


class SoftmaxRegression(_BaseClassifier):
    """Logistic regression classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
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
        List of floats, the average cross_entropy for each epoch.

    """
    def __init__(self, eta=0.01, epochs=50,
                 l2_lambda=0.0, minibatches=1,
                 random_seed=None,
                 zero_init_weight=False,
                 print_progress=0):

        super(SoftmaxRegression, self).__init__(print_progress=print_progress)
        self.random_seed = random_seed
        self.eta = eta
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.minibatches = minibatches
        self.zero_init_weight = zero_init_weight

    def _one_hot(self, y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(float)

    def _init_bias(self, n_features, n_classes):
        w = np.zeros((n_features, n_classes))
        b = np.zeros(n_classes)
        return w, b

    def _net_input(self, X, W, b):
        return (X.dot(W) + b)

    def _softmax(self, z):
        return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return - np.sum(np.log(output) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        return np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)

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
        if init_weights:
            self._n_classes = np.max(y) + 1
            self._n_features = X.shape[1]
            self.w_ = self._init_weights(shape=(self._n_features,
                                                self._n_classes),
                                         zero_init_weight=self.zero_init_weight,
                                         seed=self.random_seed)
            self.b_ = self._init_weights(shape=self._n_classes,
                                         zero_init_weight=self.zero_init_weight,
                                         seed=self.random_seed)
            self.cost_ = []

        n_idx = list(range(y.shape[0]))
        y_enc = self._one_hot(y, self._n_classes)

        # random seed for shuffling
        if self.random_seed:
            np.random.seed(self.random_seed)

        self.init_time_ = time()
        for i in range(self.epochs):
            if self.minibatches > 1:
                n_idx = np.random.permutation(n_idx)

            minis = np.array_split(n_idx, self.minibatches)
            for idx in minis:

                # givens:
                # w_ -> n_feat x n_classes
                # b_  -> n_classes

                # net_input, softmax and diff -> n_samples x n_classes:
                net = self._net_input(X[idx], self.w_, self.b_)
                softm = self._softmax(net)
                diff = softm - y_enc[idx]

                # gradient -> n_features x n_classes
                grad = np.dot(X[idx].T, diff)

                # update in opp. direction of the cost gradient
                self.w_ -= (self.eta * grad +
                            self.eta * self.l2_lambda * self.w_)
                self.b_ -= np.mean(diff, axis=0)

            # compute cost of the whole epoch
            net = self._net_input(X, self.w_, self.b_)
            softm = self._softmax(net)
            cross_ent = self._cross_entropy(output=softm, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)

            if self.print_progress:
                self._print_progress(epoch=i+1, cost=cost)

        return self

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class probabilties : array-like, shape= [n_samples, n_classes]

        """
        net = self._net_input(X, self.w_, self.b_)
        softm = self._softmax(net)
        return softm

    def predict(self, X):
        """Predict class labels of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class_labels : array-like, shape = [n_samples]
          Predicted class labels.

        """
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
