# Sebastian Raschka 2015
# mlxtend Machine Learning Library Extensions

import numpy as np

class Adaline(object):
    """ ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)

    epochs : int
      Passes over the training dataset.

    learning : str (default: sgd)
      Gradient decent (gd) or stochastic gradient descent (sgd)

    shuffle : bool (default: False)
        Shuffles training data every epoch if True to prevent circles.

    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.

    zero_init_weight : bool (default: False)
        If True, weights are initialized to zero instead of small random
        numbers in the interval [0,1]

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.

    cost_ : list
      Sum of squared errors after each epoch.

    """
    def __init__(self, eta=0.01, epochs=50,  learning='sgd',
                 random_seed=None, shuffle=False, zero_init_weight=False):

        np.random.seed(random_seed)
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle

        if learning not in ('gd', 'sgd'):
            raise ValueError('learning must be gd or sgd')
        self.learning = learning
        self.zero_init_weight = zero_init_weight

    def fit(self, X, y, init_weights=True):
        """ Fit training data.

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
        # check array shape
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:,np.newaxis]')

        # check if {0, 1} or {-1, 1} class labels are used
        self.classes_ = np.unique(y)
        if not len(self.classes_) == 2 \
                or not self.classes_[0] in (-1, 0) \
                or not self.classes_[1] == 1:
            raise ValueError('Only supports binary class labels {0, 1} or {-1, 1}.')
        if self.classes_[0] == -1:
            self.thres_ = 0.0
        else:
            self.thres_ = 0.5

        # initialize weights
        if not isinstance(init_weights, np.ndarray):
            if self.zero_init_weight:
                self.w_ = np.zeros(1 + X.shape[1])
            else:
                self.w_ = np.random.ranf(1 + X.shape[1])
        else:
            self.w_ = init_weights
        self.w_.astype('int64')

        self.cost_ = []

        for _ in range(self.epochs):

            if self.shuffle:
                X, y = self._shuffle(X, y)

            if self.learning == 'gd':
                y_val = self.net_input(X)
                errors = (y - y_val)
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = (errors**2).sum() / 2.0

            elif self.learning == 'sgd':
                cost = 0.0
                for xi, yi in zip(X, y):
                    yi_val = self.net_input(xi)
                    error = (yi - yi_val)
                    self.w_[1:] += self.eta * xi.dot(error)
                    self.w_[0] += self.eta * error
                    cost += error**2 / 2.0
            self.cost_.append(cost)

        return self

    def _shuffle(self, X, y):
        """ Unison shuffling """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X):
        """ Net input function """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Activation function """
        return self.net_input(X)

    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class : int
          Predicted class label.

        """
        return np.where(self.net_input(X) >= self.thres_, self.classes_[1], self.classes_[0])