## LogisticRegression



*LogisticRegression(eta=0.01, epochs=50, regularization=None, l2_lambda=0.0, learning='sgd', shuffle=False, random_seed=None, zero_init_weight=False)*

Logistic regression classifier.

**Parameters**


- `eta` : float (default: 0.01)

    Learning rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Passes over the training dataset.

- `learning` : str (default: sgd)

    Learning rule, sgd (stochastic gradient descent)
    or gd (gradient descent).

- `regularization` : {None, 'l2'} (default: None)

    Type of regularization. No regularization if
    `regularization=None`.

- `l2_lambda` : float

    Regularization parameter for L2 regularization.
    No regularization if l2_lambda=0.0.

- `shuffle` : bool (default: False)

    Shuffles training data every epoch if True to prevent circles.

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights.

- `zero_init_weight` : bool (default: False)

    If True, weights are initialized to zero instead of small random
    numbers in the interval [-0.1, 0.1];
    ignored if solver='normal equation'

**Attributes**


- `w_` : 1d-array

    Weights after fitting.


- `cost_` : list

    List of floats with sum of squared error cost (sgd or gd) for every
    epoch.

### Methods



*activation(X)*

Predict class probabilities of X from the net input.

**Parameters**


- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**


- `Class 1 probability` : float


*fit(X, y, init_weights=True)*

Learn weight coefficients from training data.

**Parameters**


- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values.

- `init_weights` : bool (default: True)

    (Re)initializes weights to small random floats if True.

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


- `class` : int

    Predicted class label(s).