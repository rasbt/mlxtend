## PredefinedHoldoutSplit

*PredefinedHoldoutSplit(valid_indices)*

Train/Validation set splitter for sklearn's GridSearchCV etc.

Uses user-specified train/validation set indices to split a dataset
into train/validation sets using user-defined or random
indices.

**Parameters**

- `valid_indices` : array-like, shape (num_examples,)

    Indices of the training examples in the training set
    to be used for validation. All other indices in the
    training set are used to for a training subset
    for model fitting.

### Methods

<hr>

*get_n_splits(X=None, y=None, groups=None)*

Returns the number of splitting iterations in the cross-validator

**Parameters**

- `X` : object

    Always ignored, exists for compatibility.


- `y` : object

    Always ignored, exists for compatibility.


- `groups` : object

    Always ignored, exists for compatibility.

**Returns**

- `n_splits` : 1

    Returns the number of splitting iterations in the cross-validator.
    Always returns 1.

<hr>

*split(X, y, groups=None)*

Generate indices to split data into training and test set.

**Parameters**

- `X` : array-like, shape (num_examples, num_features)

    Training data, where num_examples is the number of examples
    and num_features is the number of features.


- `y` : array-like, shape (num_examples,)

    The target variable for supervised learning problems.
    Stratification is done based on the y labels.


- `groups` : object

    Always ignored, exists for compatibility.

**Yields**

- `train_index` : ndarray

    The training set indices for that split.


- `valid_index` : ndarray

    The validation set indices for that split.

