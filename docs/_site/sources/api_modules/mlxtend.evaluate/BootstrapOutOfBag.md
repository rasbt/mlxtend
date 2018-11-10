## BootstrapOutOfBag

*BootstrapOutOfBag(n_splits=200, random_seed=None)*

**Parameters**


- `n_splits` : int (default=200)

    Number of bootstrap iterations.
    Must be larger than 1.


- `random_seed` : int (default=None)

    If int, random_seed is the seed used by
    the random number generator.


**Returns**

- `train_idx` : ndarray

    The training set indices for that split.


- `test_idx` : ndarray

    The testing set indices for that split.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/](http://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/)

### Methods

<hr>

*get_n_splits(X=None, y=None, groups=None)*

Returns the number of splitting iterations in the cross-validator

**Parameters**

- `X` : object

    Always ignored, exists for compatibility with scikit-learn.


- `y` : object

    Always ignored, exists for compatibility with scikit-learn.


- `groups` : object

    Always ignored, exists for compatibility with scikit-learn.

**Returns**


- `n_splits` : int

    Returns the number of splitting iterations in the cross-validator.

<hr>

*split(X, y=None, groups=None)*

y : array-like or None (default: None)
Argument is not used and only included as parameter
for compatibility, similar to `KFold` in scikit-learn.


- `groups` : array-like or None (default: None)

    Argument is not used and only included as parameter
    for compatibility, similar to `KFold` in scikit-learn.

