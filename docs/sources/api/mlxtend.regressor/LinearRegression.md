## LinearRegression



*LinearRegression(solver='normal equation', eta=0.01, epochs=50, random_seed=None, shuffle=False, zero_init_weight=False)*

Ordinary least squares linear regression.

**Parameters**


- `solver` : {'gd', 'sgd', 'normal equation'} (default: 'normal equation')

    Method for solving the cost function. 'gd' for gradient descent,
    'sgd' for stochastic gradient descent, or 'normal equation' (default)
    to solve the cost function analytically.

- `eta` : float (default: 0.1)

    Learning rate (between 0.0 and 1.0);
    ignored if solver='normal equation'.

- `epochs` : int (default: 50)

    Passes over the training dataset;
    ignored if solver='normal equation'.

- `shuffle` : bool (default: False)

    Shuffles training data every epoch if True to prevent circles;
    ignored if solver='normal equation'.

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights;
    ignored if solver='normal equation'.

- `zero_init_weight` : bool (default: False)

    If True, weights are initialized to zero instead of small random
    numbers in the interval [-0.1, 0.1];
    ignored if solver='normal equation'

**Attributes**


- `w_` : 1d-array

    Weights after fitting.

- `cost_` : list

    Sum of squared errors after each epoch;
    ignored if solver='normal equation'

### Methods



*fit(X, y, init_weights=True)*

Learn weight coefficients from training data.

**Parameters**


- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values.

- `init_weights` : bool (default: True)

    Re-initializes weights prior to fitting. Set False to continue
    training with weights from a previous fitting.

**Returns**


- `self` : object


*net_input(X)*

Compute the linear net input.

*predict(X)*

Predict class labels of X.

**Parameters**


- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**


- `float` : Predicted target value.
