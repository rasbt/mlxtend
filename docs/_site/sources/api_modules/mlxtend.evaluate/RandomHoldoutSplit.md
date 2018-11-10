## RandomHoldoutSplit

*RandomHoldoutSplit(valid_size=0.5, random_seed=None, stratify=False)*

Train/Validation set splitter for sklearn's GridSearchCV etc.

Provides train/validation set indices to split a dataset
into train/validation sets using random indices.

**Parameters**

- `valid_size` : float (default: 0.5)

    Proportion of examples that being assigned as
    validation examples. 1-`valid_size` will then automatically
    be assigned as training set examples.

- `random_seed` : int (default: None)

    The random seed for splitting the data
    into training and validation set partitions.

- `stratify` : bool (default: False)

    True or False, whether to perform a stratified
    split or not

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

    Training data, where num_examples is the number of
    training examples and num_features is the number of features.


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

