# Shuffle Arrays in Unison

A function for NumPy arrays in unison.

> from mlxtend.preprocessing import shuffle_arrays_unison

# Examples

## Example 1 - Scaling a Pandas DataFrame


```python
import numpy as np
from mlxtend.preprocessing import shuffle_arrays_unison
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, 3])
print('X:\n%s' % X)
print('y:\n%s' % y)
```

    X:
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    y:
    [1 2 3]



```python
X2, y2 = shuffle_arrays_unison(arrays=[X, y], random_state=3)
print('X2:\n%s' % X2)
print('y2:\n%s' % y2)
```

    X2:
    [[4 5 6]
     [1 2 3]
     [7 8 9]]
    y2:
    [2 1 3]


# API


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


