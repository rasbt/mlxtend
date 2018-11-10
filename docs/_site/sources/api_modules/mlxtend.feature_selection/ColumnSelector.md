## ColumnSelector

*ColumnSelector(cols=None, drop_axis=False)*

Object for selecting specific columns from a data set.

**Parameters**

- `cols` : array-like (default: None)

    A list specifying the feature indices to be selected. For example,
    [1, 4, 5] to select the 2nd, 5th, and 6th feature columns.
    If None, returns all columns in the array.


- `drop_axis` : bool (default=False)

    Drops last axis if True and the only one column is selected. This
    is useful, e.g., when the ColumnSelector is used for selecting
    only one column and the resulting array should be fed to e.g.,
    a scikit-learn column selector. E.g., instead of returning an
    array with shape (n_samples, 1), drop_axis=True will return an
    aray with shape (n_samples,).

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/)

### Methods

<hr>

*fit(X, y=None)*

Mock method. Does nothing.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

self

<hr>

*fit_transform(X, y=None)*

Return a slice of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_slice` : shape = [n_samples, k_features]

    Subset of the feature space where k_features <= n_features

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X, y=None)*

Return a slice of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_slice` : shape = [n_samples, k_features]

    Subset of the feature space where k_features <= n_features

