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

    random_state : int
      Random state for initializing random weights.

    learning : str (default: sgd)
      Gradient decent (gd) or stochastic gradient descent (sgd)

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.

    cost_ : list
      Sum of squared errors after each epoch.

    """
    def __init__(self, eta=0.01, epochs=50, random_state=1, learning='sgd'):

        np.random.seed(random_state)
        self.eta = eta
        self.epochs = epochs

        if not learning in ('gd', 'sgd'):
            raise ValueError('learning must be gd or sgd')
        self.learning = learning

    def fit(self, X, y, init_weights=None):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        init_weights : array-like, shape = [n_features + 1]
            Initial weights for the classifier. If None, weights
            are initialized to 0.

        Returns
        -------
        self : object
        """
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:,np.newaxis]')

        if not (np.unique(y) == np.array([-1,  1])).all():
            raise ValueError('Supports only binary class labels -1 and 1')

        if not isinstance(init_weights, np.ndarray):
            self.w_ = np.random.random(1 + X.shape[1])
        else:
            self.w_ = init_weights

        self.cost_ = []

        for i in range(self.epochs):

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
        return np.where(self.activation(X) >= 0.0, 1, -1)