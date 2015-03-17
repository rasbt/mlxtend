# Sebastian Raschka 2015
# mlxtend Machine Learning Library Extensions


import numpy as np

class Perceptron(object):
    """Perceptron classifier.
    
    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)

    epochs : int
      Passes over the training dataset.

    random_state : int
      Random state for initializing random weights.

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.

    cost_ : list
      Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.1, epochs=50, random_state=1):

        np.random.seed(random_state)
        self.eta = eta
        self.epochs = epochs

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
            self.thres_ = 0
        else:
            self.thres_ = 0.5

        # initialize weights
        if not isinstance(init_weights, np.ndarray):
            self.w_ = np.random.random(1 + X.shape[1])
        else:
            self.w_ = init_weights

        self.cost_ = []

        # learn weights
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            self.cost_.append(errors)
        return self

    def net_input(self, X):
        """ Net input function """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """ Predict class labels for X.
        
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