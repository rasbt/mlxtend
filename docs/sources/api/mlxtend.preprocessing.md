## DenseTransformer



*DenseTransformer(some_param=True)*

Convert a sparse matrix into a dense matrix.

### Methods



*fit(X, y=None)*

None

*fit_transform(X, y=None)*

None

*get_params(deep=True)*

None

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



*fit(X)*

None

*fit_transform(X)*

None

*transform(X)*

None

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



*shuffle_arrays_unison(arrays, random_state=None)*

Shuffle NumPy arrays in unison.

**Parameters**


- `arrays` : array-like, shape = [n_arrays]

    A list of NumPy arrays.

- `random_state` : int (default: None)

    Sets the random seed.

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



*standardize(array, columns, ddof=0)*

standardize columns in pandas DataFrames.

**Parameters**


- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].



- `columns` : array-like, shape = [n_columns]

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]


- `ddof` : int (default: 0)

    Delta Degrees of Freedom. The divisor used in calculations
    is N - ddof, where N represents the number of elements.

**Returns**


- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with standardized columns.

## TransformerObj



*TransformerObj()*

None

### Methods



