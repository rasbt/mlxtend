# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Implementation of a Multi-layer Perceptron in Tensorflow
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from time import time

import numpy as np
from scipy.special import expit

from .._base import _BaseModel, _Classifier, _IterativeModel, _MultiClass, _MultiLayer


class MultiLayerPerceptron(
    _BaseModel, _IterativeModel, _MultiClass, _MultiLayer, _Classifier
):
    """Multi-layer perceptron classifier with logistic sigmoid activations

    Parameters
    ------------
    eta : float (default: 0.5)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    hidden_layers : list (default: [50])
        Number of units per hidden layer. By default 50 units in the
        first hidden layer. At the moment only 1 hidden layer is supported
    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.
    l1 : float (default: 0.0)
        L1 regularization strength
    l2 : float (default: 0.0)
        L2 regularization strength
    momentum : float (default: 0.0)
        Momentum constant. Factor multiplied with the
        gradient of the previous epoch t-1 to improve
        learning speed
        w(t) := w(t) - (grad(t) + momentum * grad(t-1))
    decrease_const : float (default: 0.0)
        Decrease constant. Shrinks the learning rate
        after each epoch via eta / (1 + epoch*decrease_const)
    minibatches : int (default: 1)
        Divide the training data into *k* minibatches
        for accelerated stochastic gradient descent learning.
        Gradient Descent Learning if `minibatches` = 1
        Stochastic Gradient Descent learning if `minibatches` = len(y)
        Minibatch learning if `minibatches` > 1
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
    w_ : 2d-array, shape=[n_features, n_classes]
        Weights after fitting.
    b_ : 1D-array, shape=[n_classes]
        Bias units after fitting.
    cost_ : list
        List of floats; the mean categorical cross entropy
        cost after each epoch.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/classifier/MultiLayerPerceptron/

    """

    def __init__(
        self,
        eta=0.5,
        epochs=50,
        hidden_layers=[50],
        n_classes=None,
        momentum=0.0,
        l1=0.0,
        l2=0.0,
        dropout=1.0,
        decrease_const=0.0,
        minibatches=1,
        random_seed=None,
        print_progress=0,
    ):
        _BaseModel.__init__(self)
        _Classifier.__init__(self)
        _IterativeModel.__init__(self)
        _MultiClass.__init__(self)
        _MultiLayer.__init__(self)

        if len(hidden_layers) > 1:
            raise AttributeError("Currently, only 1 hidden layer is supported")
        self.hidden_layers = hidden_layers
        self.eta = eta
        self.n_classes = n_classes
        self.l1 = l1
        self.l2 = l2
        self.decrease_const = decrease_const
        self.momentum = momentum
        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True):
        self._check_target_array(y)

        if init_params:
            self._decr_eta = self.eta
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1

            self._n_features = X.shape[1]
            self._weight_maps, self._bias_maps = self._layermapping(
                n_features=self._n_features,
                n_classes=self.n_classes,
                hidden_layers=self.hidden_layers,
            )

            self.w_, self.b_ = self._init_params_from_layermapping(
                weight_maps=self._weight_maps,
                bias_maps=self._bias_maps,
                random_seed=self.random_seed,
            )

            self.cost_ = []

            if self.momentum != 0.0:
                prev_grad_b_1 = np.zeros(shape=self.b_["1"].shape)
                prev_grad_w_1 = np.zeros(shape=self.w_["1"].shape)
                prev_grad_b_out = np.zeros(shape=self.b_["out"].shape)
                prev_grad_w_out = np.zeros(shape=self.w_["out"].shape)

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float64)

        self.init_time_ = time()

        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                rgen=rgen, n_batches=self.minibatches, data_ary=y, shuffle=True
            ):
                net_1, act_1, net_out, act_out = self._feedforward(X[idx])

                # GRADIENTS VIA BACKPROPAGATION

                # [n_samples, n_classlabels]
                sigma_out = act_out - y_enc[idx]

                # [n_samples, n_hidden]
                sigmoid_derivative_1 = act_1 * (1.0 - act_1)

                # [n_samples, n_classlabels] dot [n_classlabels, n_hidden]
                # -> [n_samples, n_hidden]
                sigma_1 = np.dot(sigma_out, self.w_["out"].T) * sigmoid_derivative_1

                # [n_features, n_samples] dot [n_samples, n_hidden]
                # -> [n_features, n_hidden]
                grad_W_1 = np.dot(X[idx].T, sigma_1)

                grad_B_1 = np.sum(sigma_1, axis=0)

                # [n_hidden, n_samples] dot [n_samples, n_classlabels]
                # -> [n_hidden, n_classlabels]
                grad_W_out = np.dot(act_1.T, sigma_out)

                grad_B_out = np.sum(sigma_out, axis=0)

                # LEARNING RATE ADJUSTEMENTS
                self._decr_eta /= 1.0 + self.decrease_const * i

                # REGULARIZATION AND WEIGHT UPDATES

                dW_1 = (
                    self._decr_eta * grad_W_1 + self._decr_eta * self.l2 * self.w_["1"]
                )

                dW_out = (
                    self._decr_eta * grad_W_out
                    + self._decr_eta * self.l2 * self.w_["out"]
                )

                dB_1 = self._decr_eta * grad_B_1
                dB_out = self._decr_eta * grad_B_out

                self.w_["1"] -= dW_1
                self.b_["1"] -= dB_1
                self.w_["out"] -= dW_out
                self.b_["out"] -= dB_out

                if self.momentum != 0.0:
                    self.w_["1"] -= self.momentum * prev_grad_w_1
                    self.b_["1"] -= self.momentum * prev_grad_b_1
                    self.w_["out"] -= self.momentum * prev_grad_w_out
                    self.b_["out"] -= self.momentum * prev_grad_b_out
                    prev_grad_b_1 = grad_B_1
                    prev_grad_w_1 = grad_W_1
                    prev_grad_b_out = grad_B_out
                    prev_grad_w_out = grad_W_out

            net_1, act_1, net_out, act_out = self._feedforward(X)
            cross_ent = self._cross_entropy(output=act_out, y_target=y_enc)
            cost = self._compute_cost(cross_ent)

            self.cost_.append(cost)
            if self.print_progress:
                self._print_progress(iteration=i + 1, n_iter=self.epochs, cost=cost)

        return self

    def _feedforward(self, X):
        # [n_samples, n_features] dot [n_features, n_hidden]
        # -> [n_samples, n_hidden]
        net_1 = np.dot(X, self.w_["1"]) + self.b_["1"]
        act_1 = self._sigmoid(net_1)

        # [n_samples, n_hidden] dot [n_hidden, n_classlabels]
        # -> [n_samples, n_classlabels]
        net_out = np.dot(act_1, self.w_["out"]) + self.b_["out"]
        act_out = self._softmax(net_out)

        return net_1, act_1, net_out, act_out

    def _compute_cost(self, cross_entropy):
        L2_term = self.l2 * (
            np.sum(self.w_["1"] ** 2.0) + np.sum(self.w_["out"] ** 2.0)
        )

        L1_term = self.l1 * (np.abs(self.w_["1"]).sum() + np.abs(self.w_["out"]).sum())

        cross_entropy = cross_entropy + L2_term + L1_term
        return 0.5 * np.mean(cross_entropy)

    def _predict(self, X):
        net_1, act_1, net_out, act_out = self._feedforward(X)
        y_pred = np.argmax(net_out, axis=1)
        return y_pred

    def _softmax(self, z):
        e_x = np.exp(z - z.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out
        # return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

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
        net_1, act_1, net_out, act_out = self._feedforward(X)
        softm = self._softmax(act_out)
        return softm

    def _sigmoid(self, z):
        """Compute logistic function (sigmoid).
        Uses scipy.special.expit to avoid overflow
        error for very small input values z.
        """
        # return 1.0 / (1.0 + np.exp(-z))
        return expit(z)
