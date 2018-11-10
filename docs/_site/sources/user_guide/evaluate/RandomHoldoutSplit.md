# RandomHoldoutSplit

Randomly split a dataset into a train and validation subset for validation.

> `from mlxtend.evaluate import RandomHoldoutSplit`    

## Overview

The `RandomHoldoutSplit` class serves as an alternative to scikit-learn's `KFold` class, where the `RandomHoldoutSplit` class splits a dataset into training and a validation subsets without rotation. The `RandomHoldoutSplit` can be used as argument for `cv` parameters in scikit-learn's `GridSearchCV` etc.

The term "random" in `RandomHoldoutSplit` comes from the fact that the split is specified by the `random_seed` rather than specifying the training and validation set indices manually as in the `PredefinedHoldoutSplit` class in mlxtend.

## Example 1 -- Iterating Over a RandomHoldoutSplit


```python
from mlxtend.evaluate import RandomHoldoutSplit
from mlxtend.data import iris_data

X, y = iris_data()
h_iter = RandomHoldoutSplit(valid_size=0.3, random_seed=123)

cnt = 0
for train_ind, valid_ind in h_iter.split(X, y):
    cnt += 1
    print(cnt)
```

    1



```python
print(train_ind[:5])
print(valid_ind[:5])
```

    [ 60  16  88 130   6]
    [ 72 125  80  86 117]


## Example 2 -- RandomHoldoutSplit in GridSearch


```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.evaluate import RandomHoldoutSplit
from mlxtend.data import iris_data

X, y = iris_data()

params = {'n_neighbors': [1, 2, 3, 4, 5]}

grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid=params,
                    cv=RandomHoldoutSplit(valid_size=0.3, random_seed=123))

grid.fit(X, y)

assert grid.n_splits_ == 1
print(grid.grid_scores_)
```

    [mean: 0.95556, std: 0.00000, params: {'n_neighbors': 1}, mean: 0.95556, std: 0.00000, params: {'n_neighbors': 2}, mean: 0.95556, std: 0.00000, params: {'n_neighbors': 3}, mean: 0.95556, std: 0.00000, params: {'n_neighbors': 4}, mean: 0.95556, std: 0.00000, params: {'n_neighbors': 5}]


    /Users/sebastian/miniconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:762: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)


## API


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


