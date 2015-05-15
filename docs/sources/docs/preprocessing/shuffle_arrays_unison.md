mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>
# Array Unison Shuffling

> from mlxtend.preprocessing import shuffle_arrays_unison

A function that shuffles 2 or more NumPy arrays in unison.

<hr>
    
## Example


	>>> import numpy as np
    >>> from mlxtend.preprocessing import shuffle_arrays_unison
    >>> X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y1 = np.array([1, 2, 3])
    >>> print(X1)
    [[1 2 3]
    [4 5 6]
    [7 8 9]]    
    >>> print(y1)
    [1 2 3]
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)
    >>> print(X2)
    [[4 5 6]
    [1 2 3]
    [7 8 9]]
    >>> print(y1)
    [2 1 3]

## Default Parameters

<pre>def shuffle_arrays_unison(arrays, random_state=None):
    """
    Shuffle NumPy arrays in unison.

    Parameters
    ----------
    arrays : array-like, shape = [n_arrays]
      A list of NumPy arrays.

    random_state : int
      Sets the random state.

    Returns
    ----------
    shuffled_arrays : A list of NumPy arrays after shuffling.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxtend.preprocessing import shuffle_arrays_unison
    >>> X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y1 = np.array([1, 2, 3])
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_state=3)
    >>> assert(X2.all() == np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]).all())
    >>> assert(y2.all() == np.array([2, 1, 3]).all())
    >>>
    """</pre>