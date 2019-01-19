# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.utils import check_Xy


def shuffle_arrays_unison(arrays, random_seed=None):
    """Shuffle NumPy arrays in unison.

    Parameters
    ----------
    arrays : array-like, shape = [n_arrays]
        A list of NumPy arrays.
    random_seed : int (default: None)
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
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)
    >>> assert(X2.all() == np.array([[4, 5, 6], [1, 2, 3], [7, 8, 9]]).all())
    >>> assert(y2.all() == np.array([2, 1, 3]).all())
    >>>

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/shuffle_arrays_unison/
    """
    if random_seed:
        np.random.seed(random_seed)
    n = len(arrays[0])
    for a in arrays:
        assert(len(a) == n)
    idx = np.random.permutation(n)
    return [a[idx] for a in arrays]


def shuffled_split(X, y, shuffle=True, train_size=0.75, random_seed=None):
    """Splits feature and target arrays into training and test subsets.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Initial dataset, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape = [n_samples]
        Target values.
    shuffle : bool (default: True)
        Doesn't shuffle the arrays if False
    train_size : float (default: 0.75)
        Proportion of data in the training arrays. For example, 0.75 will
        put 75% of the data into the training array, and 25% of the data
        into the test array.
    random_seed : int (default: None)
        Sets the random state.

    Returns
    ----------
    X_train : array-like, shape = [n_samples * train_size, n_features]
        Training dataset, where n_samples is the number of samples and
        n_features is the number of features.
    y_train : array-like, shape = [n_samples * train_size]
        Training target values.
    X_test : array-like, shape = [n_samples * (1-train_size), n_features]
        Dataset for testing, where n_samples is the number of samples and
        n_features is the number of features.
    y_test : array-like, shape = [n_samples * (1-train_size)]
         Target values for testing.

    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/preprocessing/shuffled_split/

    """
    check_Xy(X, y, y_int=False)

    if train_size <= 0.0 or train_size >= 1.0:
        raise ValueError('train_size must be a float in the range (0.0, 1.0)')

    if shuffle:
        X_ary, y_ary = shuffle_arrays_unison(arrays=[X.copy(), y.copy()],
                                             random_seed=random_seed)
    else:
        X_ary, y_ary = X.copy(), y.copy()

    train_absize = round(train_size * y.shape[0])

    X_train, y_train = X_ary[:train_absize], y_ary[:train_absize]
    X_test, y_test = X_ary[train_absize:], y_ary[train_absize:]
    return X_train, y_train, X_test, y_test
