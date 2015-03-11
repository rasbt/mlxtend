# Sebastian Raschka 2015
# mlxtend Machine Learning Library Extensions


import numpy as np

class Perceptron(object):
    """Perceptron Classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between >0.0 and <1.0)

    epochs : int
      Passes over the training dataset.

    learning : str (default: sgd)
      Learning rule, sgd (stochastic gradient descent),
      gd (gradient descent) or perceptron (Rosenblatt's perceptron rule).


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.

    cost_ : list
      List of floats with sum of squared error cost (sgd or gd)
      or number of misclassifications (perceptron) for every
      epoch.

    """
    def __init__(self, eta=0.01, epochs=50, learning='sgd'):
        self.eta = eta
        self.epochs = epochs

        if not learning in ('sgd', 'gd', 'perceptron'):
            raise ValueError('learning must be sgd, gd, or perceptron')
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

        if (np.unique(y) != np.array([-1,  1])).all() \
                or (np.unique(y) != np.array([0,  1])).all():
            raise ValueError('Supports only binary class labels -1, 1 or 0, 1')

        self.c1_, self.c2_ = np.unique(y)
        if self.c1_ == -1:
            self.t_ = 0.0
        elif self.c1_ == 0:
            self.t_ = 0.5


        if not isinstance(init_weights, np.ndarray):
        # Initialize weights to 0
            self.w_ = np.zeros(1 + X.shape[1])
        else:
            self.w_ = init_weights

        self.cost_ = []

        for i in range(self.epochs):

            if self.learning == 'gd':
                y_pred = self.activate(X)
                errors = (y - y_pred)
                self.w_[1:] += self.eta * X.T.dot(errors)
                self.w_[0] += self.eta * errors.sum()
                cost = (errors**2).sum() / 2.0

            elif self.learning == 'sgd':
                cost = 0.0
                for xi, yi in zip(X, y):
                    yi_pred = self.activate(xi)
                    error = (yi - yi_pred)
                    self.w_[1:] += self.eta * xi.dot(error)
                    self.w_[0] += self.eta * error
                    cost += error**2
                cost /= 2.0

            elif self.learning == 'perceptron':
                cost = 0.0
                for xi, yi in zip(X, y):
                    yi_pred = self.predict(xi)
                    error = (yi - yi_pred)
                    self.w_[1:] += self.eta * xi.dot(error)
                    self.w_[0] += self.eta * error
                    cost += int(yi_pred != yi)

            self.cost_.append(cost)
        return self

    def activate(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        int
          Raw value that can be thresholded to predict the class label.

        """
        return np.dot(X, self.w_[1:]) + self.w_[0]

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
        return np.where(self.activate(X) >= self.t_, self.c2_, self.c1_)
