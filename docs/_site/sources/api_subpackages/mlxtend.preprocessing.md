mlxtend version: 0.15.0dev 
## CopyTransformer

*CopyTransformer()*

Transformer that returns a copy of the input array

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/)

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

Return a copy of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_copy` : copy of the input X array.


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

Return a copy of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_copy` : copy of the input X array.





## DenseTransformer

*DenseTransformer(return_copy=True)*

Convert a sparse array into a dense array.

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/DenseTransformer/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/DenseTransformer/)

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

Return a dense version of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_dense` : dense version of the input X array.


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

Return a dense version of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_dense` : dense version of the input X array.





## MeanCenterer

*MeanCenterer()*

Column centering of vectors and matrices.

**Attributes**

- `col_means` : numpy.ndarray [n_columns]

    NumPy array storing the mean values for centering after fitting
    the MeanCenterer object.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/preprocessing/MeanCenterer/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/MeanCenterer/)

### Methods

<hr>

*fit(X)*

Gets the column means for mean centering.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Array of data vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

self

<hr>

*fit_transform(X)*

Fits and transforms an arry.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Array of data vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `X_tr` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    A copy of the input array with the columns centered.

<hr>

*transform(X)*

Centers a NumPy array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Array of data vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `X_tr` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    A copy of the input array with the columns centered.




## OnehotTransactions

*OnehotTransactions(*args, **kwargs)*

Encoder class for transaction data in Python lists

**Parameters**

None

**Attributes**

columns_: list
List of unique names in the `X` input list of lists

**Examples**

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/)

### Methods

<hr>

*fit(X)*

Learn unique column names from transaction DataFrame

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

<hr>

*fit_transform(X, sparse=False)*

Fit a TransactionEncoder encoder and transform a dataset.

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

*inverse_transform(array)*

Transforms an encoded NumPy array back into transactions.

**Parameters**

- `array` : NumPy array [n_transactions, n_unique_items]

    The NumPy one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order

    For example,
```
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
```
    The corresponding column labels are available as self.columns_,
    e.g., ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

**Returns**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
```
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]
```

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

*transform(X, sparse=False)*

Transform transactions into a one-hot encoded NumPy array.

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

    sparse: bool (default=False)
    If True, transform will return Compressed Sparse Row matrix
    instead of the regular one.

**Returns**

- `array` : NumPy array [n_transactions, n_unique_items]

    if sparse=False (default).
    Compressed Sparse Row matrix otherwise
    The one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order. Exact representation depends
    on the sparse argument

    For example,
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
    The corresponding column labels are available as self.columns_, e.g.,
    ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']




## TransactionEncoder

*TransactionEncoder()*

Encoder class for transaction data in Python lists

**Parameters**

None

**Attributes**

columns_: list
List of unique names in the `X` input list of lists

**Examples**

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/)

### Methods

<hr>

*fit(X)*

Learn unique column names from transaction DataFrame

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

<hr>

*fit_transform(X, sparse=False)*

Fit a TransactionEncoder encoder and transform a dataset.

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

*inverse_transform(array)*

Transforms an encoded NumPy array back into transactions.

**Parameters**

- `array` : NumPy array [n_transactions, n_unique_items]

    The NumPy one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order

    For example,
```
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
```
    The corresponding column labels are available as self.columns_,
    e.g., ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

**Returns**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
```
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]
```

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

*transform(X, sparse=False)*

Transform transactions into a one-hot encoded NumPy array.

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

    sparse: bool (default=False)
    If True, transform will return Compressed Sparse Row matrix
    instead of the regular one.

**Returns**

- `array` : NumPy array [n_transactions, n_unique_items]

    if sparse=False (default).
    Compressed Sparse Row matrix otherwise
    The one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order. Exact representation depends
    on the sparse argument

    For example,
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
    The corresponding column labels are available as self.columns_, e.g.,
    ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']




## minmax_scaling

*minmax_scaling(array, columns, min_val=0, max_val=1)*

Min max scaling of pandas' DataFrames.

**Parameters**

- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].


- `columns` : array-like, shape = [n_columns]

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]

- `min_val` : `int` or `float`, optional (default=`0`)

    minimum value after rescaling.

- `max_val` : `int` or `float`, optional (default=`1`)

    maximum value after rescaling.

**Returns**

- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with rescaled columns.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/preprocessing/minmax_scaling/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/minmax_scaling/)




## one_hot

*one_hot(y, num_labels='auto', dtype='float')*

One-hot encoding of class labels

**Parameters**

- `y` : array-like, shape = [n_classlabels]

    Python list or numpy array consisting of class labels.

- `num_labels` : int or 'auto'

    Number of unique labels in the class label array. Infers the number
    of unique labels from the input array if set to 'auto'.

- `dtype` : str

    NumPy array type (float, float32, float64) of the output array.

**Returns**

- `ary` : numpy.ndarray, shape = [n_classlabels]

    One-hot encoded array, where each sample is represented as
    a row vector in the returned array.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/)




## shuffle_arrays_unison

*shuffle_arrays_unison(arrays, random_seed=None)*

Shuffle NumPy arrays in unison.

**Parameters**

- `arrays` : array-like, shape = [n_arrays]

    A list of NumPy arrays.

- `random_seed` : int (default: None)

    Sets the random state.

**Returns**

- `shuffled_arrays` : A list of NumPy arrays after shuffling.


**Examples**

    >>> import numpy as np
    >>> from mlxtend.preprocessing import shuffle_arrays_unison
    >>> X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y1 = np.array([1, 2, 3])
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)
    >>> assert(X2.all() == np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]).all())
    >>> assert(y2.all() == np.array([2, 1, 3]).all())
    >>>

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/shuffle_arrays_unison/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/shuffle_arrays_unison/)




## standardize

*standardize(array, columns=None, ddof=0, return_params=False, params=None)*

Standardize columns in pandas DataFrames.

**Parameters**

- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].


- `columns` : array-like, shape = [n_columns] (default: None)

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]
    If None, standardizes all columns.

- `ddof` : int (default: 0)

    Delta Degrees of Freedom. The divisor used in calculations
    is N - ddof, where N represents the number of elements.

- `return_params` : dict (default: False)

    If set to True, a dictionary is returned in addition to the
    standardized array. The parameter dictionary contains the
    column means ('avgs') and standard deviations ('stds') of
    the individual columns.

- `params` : dict (default: None)

    A dictionary with column means and standard deviations as
    returned by the `standardize` function if `return_params`
    was set to True. If a `params` dictionary is provided, the
    `standardize` function will use these instead of computing
    them from the current array.

**Notes**

If all values in a given column are the same, these values are all
    set to `0.0`. The standard deviation in the `parameters` dictionary
    is consequently set to `1.0` to avoid dividing by zero.

**Returns**

- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with standardized columns.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/preprocessing/standardize/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/standardize/)




