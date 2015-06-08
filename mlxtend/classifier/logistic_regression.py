# Sebastian Raschka 2015
# mlxtend Machine Learning Library Extensions


import numpy as np

class LogisticRegression(object):
    """Logistic regression classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)

    epochs : int
      Passes over the training dataset.

    learning : str (default: sgd)
      Learning rule, sgd (stochastic gradient descent)
      or gd (gradient descent).

    regularization : {None, 'l2'} (default: None)
      Type of regularization. No regularization if
      `regularization=None`.

    lambda_ : float
      Regularization parameter for L2 regularization.
      No regularization if `regularization=None`.

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
      List of floats with sum of squared error cost (sgd or gd) for every
      epoch.

    """
    def __init__(self, eta=0.01, epochs=50, regularization=None,
                 lambda_=1.0, learning='sgd', shuffle=False,
                 random_seed=None, zero_init_weight=False):

        np.random.seed(random_seed)
        self.eta = eta
        self.epochs = epochs
        self.lambda_ = lambda_
        self.shuffle = shuffle
        self.regularization = regularization
        self.zero_init_weight = zero_init_weight

        if learning not in ('sgd', 'gd'):
            raise ValueError('learning must be sgd or gd')
        self.learning = learning

        if self.regularization not in (None, 'l2'):
            raise AttributeError('regularization must be None or "l2"')

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
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:,np.newaxis]')

        if (np.unique(y) != np.array([0, 1])).all():
            raise ValueError('Supports only binary class labels 0 and 1')

        if not isinstance(init_weights, np.ndarray):
            if self.zero_init_weight:
                self.w_ = np.zeros(1 + X.shape[1])
            else:
                self.w_ = np.random.ranf(1 + X.shape[1])
        else:
            self.w_ = init_weights
        self.w_.astype('int64')
        self.m_ = len(self.w_)
        self.cost_ = []

        for _ in range(self.epochs):

            if self.shuffle:
                X, y = self._shuffle(X, y)

            if self.learning == 'gd':
                y_val = self.activation(X)
                errors = (y_val - y)
                self.lambda_ * self.w_[1:]
                self.w_[1:] = self._regularize() - self.eta * X.T.dot(errors)
                self.w_[0] -= self.eta * errors.sum()

            elif self.learning == 'sgd':
                for xi, yi in zip(X, y):
                    yi_val = self.activation(xi)
                    error = (yi_val - yi)
                    self.w_[1:] = self._regularize() - self.eta * xi.dot(error)
                    self.w_[0] -= self.eta * error
            self.cost_.append(self._logit_cost(y, self.activation(X)))
        return self

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
        # equivalent to np.where(self.activation(X) >= 0.5, 1, 0)
        return np.where(self.net_input(X) >= 0.0, 1, 0)


    def net_input(self, X):
        """ Net input function. """
        return X.dot(self.w_[1:]) + self.w_[0]


    def activation(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
          Class 1 probability : float

        """
        z = self.net_input(X)
        return self._sigmoid(z)

    def _shuffle(self, X, y):
        """ Unison shuffling """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _logit_cost(self, y, y_val):
        logit = -y.dot(np.log(y_val)) - ((1 - y).dot(np.log(1 - y_val)))
        return logit + self._regularize_cost()

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def _regularize(self):
        if self.regularization == 'l2':
            regularize = self.w_[1:] * (1 - self.eta * self.lambda_)
        else:
            regularize = self.w_[1:]
        return regularize

    def _regularize_cost(self):
        if self.regularization == 'l2':
            regularize = self.lambda_ * np.array([w**2 for w in self.w_[1:]], dtype='int64')
        else:
            regularize = np.zeros(self.m_ - 1)
        regularize + 0.5 * np.sum(regularize)
        return regularize
