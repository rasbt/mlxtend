# Sebastian Raschka 2015
# mlxtend Machine Learning Library Extensions

import numpy as np

class LinearRegression(object):
    """ Ordinary least squares linear regression.

    Parameters
    ------------
    
    solver : {'gd', 'sgd', 'normal_equation'} (default: 'normal_equation')
      Method for solving the cost function. 'gd' for gradient descent, 
      'sgd' for stochastic gradient descent, or 'normal_equation' (default)
      to solve the cost function analytically.
    
    eta : float (default: 0.1)
      Learning rate (between 0.0 and 1.0); 
      ignored if solver='normal_equation'.

    epochs : int (default: 50)
      Passes over the training dataset; 
      ignored if solver='normal_equation'.

    shuffle : bool (default: False)
        Shuffles training data every epoch if True to prevent circles;
        ignored if solver='normal_equation'.

    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights;
        ignored if solver='normal_equation'.

    zero_init_weight : bool (default: False)
        If True, weights are initialized to zero instead of small random
        numbers in the interval [0,1];
        ignored if solver='normal_equation'

    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.

    cost_ : list
      Sum of squared errors after each epoch;
      ignored if solver='normal_equation'

    """
    def __init__(self, solver='normal_equation', eta=0.01, 
                 epochs=50, random_seed=None, shuffle=False, 
                 zero_init_weight=False):

        np.random.seed(random_seed)
        self.eta = eta
        self.epochs = epochs
        self.shuffle = shuffle
        if solver not in ('normal_equation', 'gd', 'sgd'):
            raise ValueError('learning must be gd or sgd') 
        self.solver = solver
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

        if self.solver == 'normal_equation':
            self.w_ = self._normal_equation(X, y)
            
        # Gradient descent or stochastic gradient descent learning
        else:
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

                if self.solver == 'gd':
                    y_val = self.net_input(X)
                    errors = (y - y_val)
                    self.w_[1:] += self.eta * X.T.dot(errors)
                    self.w_[0] += self.eta * errors.sum()
                    cost = (errors**2).sum() / 2.0

                elif self.solver == 'sgd':
                    cost = 0.0
                    for xi, yi in zip(X, y):
                        yi_val = self.net_input(xi)
                        error = (yi - yi_val)
                        self.w_[1:] += self.eta * xi.dot(error)
                        self.w_[0] += self.eta * error
                        cost += error**2 / 2.0
                self.cost_.append(cost)

        return self
    
    def _normal_equation(self, X, y):
        """Solve linear regression analytically"""
        w = np.zeros(X.shape[1] + 1)
        z = np.linalg.inv(np.dot(X.T, X))
        w[1:] = np.dot(np.dot(z, X.T), y)

        # intercept
        y_pred = np.dot(X, w[1:])
        w[0] = np.mean(y) - np.mean(y_pred)
        return w

    def _shuffle(self, X, y):
        """ Unison shuffling """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def net_input(self, X):
        """ Net input function """
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Predict target values for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        float : Predicted target value.

        """
        return self.net_input(X)