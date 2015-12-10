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

    random_weights : list (default: [-0.5, 0.5])
        Min and max values for initializing the random weights.
        Initializes weights to 0 if None or False.

    Attributes
    -----------
    w_ : 1d-array
        Weights after fitting.

    cost_ : list
        Number of misclassifications in every epoch.

    """
    def __init__(self, eta=0.1, epochs=50, shuffle=False,
                 random_state=None, random_weights=False):

        np.random.seed(random_state)
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
        self.random_weights = random_weights

    def fit(self, X, y, init_weights=True):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        shuffle : bool (default: False)
            Shuffles training data every epoch if True to prevent circles.

        init_weights : bool (default: True)
            Re-initializes weights prior to fitting. Set False to continue
            training with weights from a previous fitting.

        Returns
        -------
        self : object

        """
        # check array shape
        if not len(X.shape) == 2:
            raise ValueError('X must be a 2D array. Try X[:,np.newaxis]')

        # check if {0, 1} or {-1, 1} class labels are used
        self.classes_ = np.unique(y)
        if not (np.all(np.array([0, 1]) == self.classes_) or
                    np.all(np.array([-1, 1]) == self.classes_)):
            raise ValueError('Only supports binary class labels {0, 1} or {-1, 1}.')

        if init_weights:
            self.w_ = self._initialize_weights(size=X.shape[1]+1)

        self.cost_ = []

        # learn weights
        for _ in range(self.epochs):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)

            self.cost_.append(errors)
        return self

    def _initialize_weights(self, size):
        """Initialize weights with small random numbers."""
        if self.random_weights:
            w = np.random.uniform(self.random_weights[0],
                                  self.random_weights[1],
                                  size=size)
        else:
            w = np.zeros(size)
        return w

    def _shuffle(self, X, y):
        """ Unison shuffling """
        r = np.random.permutation(len(y))
        return X[r], y[r]

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
        return np.where(self.net_input(X) >= 0.0, self.classes_[1], self.classes_[0])
