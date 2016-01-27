## NeuralNetMLP

*NeuralNetMLP(n_output, n_features, n_hidden=30, l1=0.0, l2=0.0, epochs=500, eta=0.001, alpha=0.0, decrease_const=0.0, random_weights=[-1.0, 1.0], shuffle_init=True, shuffle_epoch=True, minibatches=1, random_seed=None, print_progress=0)*

Feedforward neural network / Multi-layer perceptron classifier.

**Parameters**

- `n_output` : int

    Number of output units, should be equal to the
    number of unique class labels.

- `n_features` : int

    Number of features (dimensions) in the target dataset.
    Should be equal to the number of columns in the X array.

- `n_hidden` : int (default: 30)

    Number of hidden units.

- `l1` : float (default: 0.0)

    Lambda value for L1-regularization.
    No regularization if l1=0.0 (default)

- `l2` : float (default: 0.0)

    Lambda value for L2-regularization.
    No regularization if l2=0.0 (default)

- `epochs` : int (default: 500)

    Number of passes over the training set.

- `eta` : float (default: 0.001)

    Learning rate.

- `alpha` : float (default: 0.0)

    Momentum constant. Factor multiplied with the
    gradient of the previous epoch t-1 to improve
    learning speed
    w(t) := w(t) - (grad(t) + alpha*grad(t-1))

- `decrease_const` : float (default: 0.0)

    Decrease constant. Shrinks the learning rate
    after each epoch via eta / (1 + epoch*decrease_const)

- `random_weights` : list (default: [-1.0, 1.0])

    Min and max values for initializing the random weights.
    Initializes weights to 0 if None or False.

- `shuffle_init` : bool (default: True)

    Shuffles (a copy of the) training data before training.

- `shuffle_epoch` : bool (default: True)

    Shuffles training data before every epoch if True to prevent circles.

- `minibatches` : int (default: 1)

    Divides training data into k minibatches for efficiency.
    Normal gradient descent learning if k=1 (default).

- `random_seed` : int (default: None)

    Set random seed for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
    0: No output
    1: Epochs elapsed
    2: 1 plus time elapsed
    3: 2 plus estimated time until completion

**Attributes**

- `cost_` : list

    Sum of squared errors after each epoch.

### Methods

<hr>

*fit(X, y)*

Learn weight coefficients from training data.

**Parameters**

- `X` : array, shape = [n_samples, n_features]

    Input layer with original features.

- `y` : array, shape = [n_samples]

    Target class labels.

**Returns:**
    self

<hr>

*predict(X)*

Predict class labels

**Parameters**

- `X` : array, shape = [n_samples, n_features]

    Input layer with original features.

**Returns:**

- `y_pred` : array, shape = [n_samples]

    Predicted class labels.

