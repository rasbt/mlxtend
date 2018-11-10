# Mean Centerer

A transformer object that performs column-based mean centering on a NumPy array.

> from mlxtend.preprocessing import MeanCenterer

## Example 1 - Centering a NumPy Array

Use the `fit` method to fit the column means of a dataset (e.g., the training dataset) to a new `MeanCenterer` object. Then, call the `transform` method on the same dataset to center it at the sample mean.


```python
import numpy as np
from mlxtend.preprocessing import MeanCenterer
X_train = np.array(
                   [[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
mc = MeanCenterer().fit(X_train)
mc.transform(X_train)
```




    array([[-3., -3., -3.],
           [ 0.,  0.,  0.],
           [ 3.,  3.,  3.]])



## API


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


