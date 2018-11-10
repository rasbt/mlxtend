# PredefinedHoldoutSplit

Split a dataset into a train and validation subset for validation based on user-specified indices.

> `from mlxtend.evaluate import PredefinedHoldoutSplit`    

## Overview

The `PredefinedHoldoutSplit` class serves as an alternative to scikit-learn's `KFold` class, where the `PredefinedHoldoutSplit` class splits a dataset into training and a validation subsets without rotation, based on validation indices specified by the user. The `PredefinedHoldoutSplit` can be used as argument for `cv` parameters in scikit-learn's `GridSearchCV` etc.

For performing a random split, see the related `RandomHoldoutSplit` class.

## Example 1 -- Iterating Over a PredefinedHoldoutSplit


```python
from mlxtend.evaluate import PredefinedHoldoutSplit
from mlxtend.data import iris_data

X, y = iris_data()
h_iter = PredefinedHoldoutSplit(valid_indices=[0, 1, 99])

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

    [2 3 4 5 6]
    [ 0  1 99]


## Example 2 -- PredefinedHoldoutSplit in GridSearch


```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.evaluate import PredefinedHoldoutSplit
from mlxtend.data import iris_data

X, y = iris_data()


params = {'n_neighbors': [1, 2, 3, 4, 5]}

grid = GridSearchCV(KNeighborsClassifier(),
                    param_grid=params,
                    cv=PredefinedHoldoutSplit(valid_indices=[0, 1, 99]))

grid.fit(X, y)

assert grid.n_splits_ == 1
print(grid.grid_scores_)
```

    [mean: 1.00000, std: 0.00000, params: {'n_neighbors': 1}, mean: 1.00000, std: 0.00000, params: {'n_neighbors': 2}, mean: 1.00000, std: 0.00000, params: {'n_neighbors': 3}, mean: 1.00000, std: 0.00000, params: {'n_neighbors': 4}, mean: 1.00000, std: 0.00000, params: {'n_neighbors': 5}]


    /Users/sebastian/miniconda3/lib/python3.6/site-packages/sklearn/model_selection/_search.py:762: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)


## API


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


