mlxtend version: 0.3.0
## DenseTransformer

*DenseTransformer(some_param=True)*

Convert a sparse matrix into a dense matrix.

### Methods

<hr>

*fit(X, y=None)*

None

<hr>

*fit_transform(X, y=None)*

None

<hr>

*get_params(deep=True)*

None

<hr>

*transform(X, y=None)*

None

## MeanCenterer

*MeanCenterer()*

Column centering of vectors and matrices.

**Attributes**

- `col_means` : numpy.ndarray [n_columns]

    NumPy array storing the mean values for centering after fitting
    the MeanCenterer object.

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

- `min_val` : `int` or `float`, optional (default=`1`)

    maximum value after rescaling.

**Returns**

- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with rescaled columns.

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
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_state=3)
    >>> assert(X2.all() == np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]).all())
    >>> assert(y2.all() == np.array([2, 1, 3]).all())
    >>>

## standardize

*standardize(array, columns, ddof=0, return_params=False, params=None)*

Standardize columns in pandas DataFrames.

**Parameters**

- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].


- `columns` : array-like, shape = [n_columns]

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]

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

## TransformerObj

*TransformerObj()*

None

### Methods

