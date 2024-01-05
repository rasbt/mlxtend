# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Implementation of the mulitnomial logistic regression algorithm for
# classification.

# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from time import time

import numpy as np

from .._base import _BaseModel, _Classifier, _IterativeModel, _MultiClass


class SoftmaxRegression(_BaseModel, _IterativeModel, _Classifier, _MultiClass):

    """Softmax regression classifier.

    Parameters
    ------------
    eta : float (default: 0.01)
        Learning rate (between 0.0 and 1.0)

    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.

    l2 : float
        Regularization parameter for L2 regularization.
        No regularization if l2=0.0.

    minibatches : int (default: 1)
        The number of minibatches for gradient-based optimization.
        If 1: Gradient Descent learning
        If len(y): Stochastic Gradient Descent (SGD) online learning
        If 1 < minibatches < len(y): SGD Minibatch learning

    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.

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
        List of floats, the average cross_entropy for each epoch.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/

    """

    def __init__(
        self,
        eta=0.01,
        epochs=50,
        l2=0.0,
        minibatches=1,
        n_classes=None,
        random_seed=None,
        print_progress=0,
    ):
        _BaseModel.__init__(self)
        _IterativeModel.__init__(self)
        _Classifier.__init__(self)
        _MultiClass.__init__(self)

        self.eta = eta
        self.epochs = epochs
        self.l2 = l2
        self.minibatches = minibatches
        self.n_classes = n_classes
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _net_input(self, X):
        return X.dot(self.w_) + self.b_

    def _softmax_activation(self, z):
        e_x = np.exp(z - z.max(axis=1, keepdims=True))
        out = e_x / e_x.sum(axis=1, keepdims=True)
        return out
        # return (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T

    def _cross_entropy(self, output, y_target):
        return -np.sum(np.log(output) * (y_target), axis=1)

    def _cost(self, cross_entropy):
        L2_term = self.l2 * np.sum(self.w_**2)
        cross_entropy = cross_entropy + L2_term
        return 0.5 * np.mean(cross_entropy)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)

    def _forward(self, X):
        z = self._net_input(X)
        a = self._softmax_activation(z)
        return a

    def _backward(self, X, y_true, y_probas):
        grad_loss_wrt_out = y_true - y_probas
        # gradient -> n_features x n_classes
        grad_loss_wrt_w = -np.dot(X.T, grad_loss_wrt_out)
        grad_loss_wrt_b = -np.sum(grad_loss_wrt_out, axis=0)
        return grad_loss_wrt_w, grad_loss_wrt_b

    def _fit(self, X, y, init_params=True):
        self._check_target_array(y)
        if init_params:
            if self.n_classes is None:
                self.n_classes = np.max(y) + 1
            self._n_features = X.shape[1]

            self.b_, self.w_ = self._init_params(
                weights_shape=(self._n_features, self.n_classes),
                bias_shape=(self.n_classes,),
                random_seed=self.random_seed,
            )
            self.cost_ = []

        y_enc = self._one_hot(y=y, n_labels=self.n_classes, dtype=np.float_)

        self.init_time_ = time()
        rgen = np.random.RandomState(self.random_seed)
        for i in range(self.epochs):
            for idx in self._yield_minibatches_idx(
                rgen=rgen, n_batches=self.minibatches, data_ary=y, shuffle=True
            ):
                # net_input, softmax and diff -> n_samples x n_classes:
                y_probas = self._forward(X[idx])

                # w_ -> n_feat x n_classes
                # b_  -> n_classes
                grad_loss_wrt_w, grad_loss_wrt_b = self._backward(
                    X[idx], y_true=y_enc[idx], y_probas=y_probas
                )

                # update in opp. direction of the cost gradient
                l2_reg = self.l2 * self.w_
                self.w_ += self.eta * (-grad_loss_wrt_w - l2_reg)
                self.b_ += self.eta * -grad_loss_wrt_b

            # compute cost of the whole epoch
            y_probas = self._forward(X)
            cross_ent = self._cross_entropy(output=y_probas, y_target=y_enc)
            cost = self._cost(cross_ent)
            self.cost_.append(cost)

            if self.print_progress:
                self._print_progress(iteration=i + 1, n_iter=self.epochs, cost=cost)

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
        return self._forward(X)

    def _predict(self, X):
        probas = self.predict_proba(X)
        return self._to_classlabels(probas)
