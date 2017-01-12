# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def one_hot(y, num_labels='auto', dtype='float'):
    """One-hot encoding of class labels

    Parameters
    ----------
    y : array-like, shape = [n_classlabels]
        Python list or numpy array consisting of class labels.
    num_labels : int or 'auto'
        Number of unique labels in the class label array. Infers the number
        of unique labels from the input array if set to 'auto'.
    dtype : str
        NumPy array type (float, float32, float64) of the output array.

    Returns
    ----------
    ary : numpy.ndarray, shape = [n_classlabels]
        One-hot encoded array, where each sample is represented as
        a row vector in the returned array.

    """
    if not (num_labels == 'auto' or isinstance(num_labels, int)):
        raise AttributeError('num_labels must be an integer or "auto"')
    if isinstance(y, list):
        yt = np.asarray(y)
    else:
        yt = y
    if not len(yt.shape) == 1:
        raise AttributeError('y array must be 1-dimensional')
    if num_labels == 'auto':
        # uniq = np.unique(yt).shape[0]
        uniq = np.max(yt + 1)
    else:
        uniq = num_labels
    if uniq == 1:
        ary = np.array([[0.]], dtype=dtype)

    else:
        ary = np.zeros((len(y), uniq))
        for i, val in enumerate(y):
            ary[i, val] = 1

    return ary.astype(dtype)


class OnehotTransactions(object):
    """One-hot encoder class for transaction data in Python lists

    Parameters
    ------------
    None

    Attributes
    ------------
    columns_: list
      List of unique names in the `X` input list of lists

    """
    def __init__(self):
        return None

    def fit(self, X):
        """Learn unique column names from transaction DataFrame

        Parameters
        ------------
        X : list of lists
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

        """
        unique_items = set()
        for transaction in X:
            for item in transaction:
                unique_items.add(item)
        self.columns_ = sorted(unique_items)
        return self

    def transform(self, X):
        """Transform transactions into a one-hot encoded NumPy array.

        Parameters
        ------------
        X : list of lists
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

        Returns
        ------------
        onehot : NumPy array [n_transactions, n_unique_items]
           The NumPy one-hot encoded integer array of the input transactions,
           where the columns represent the unique items found in the input
           array in alphabetic order

           For example,
           array([[1, 0, 1, 1, 0, 1],
                  [1, 0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 0, 1, 1],
                  [0, 0, 1, 0, 1, 0],
                  [1, 1, 0, 0, 0, 0]])
          The corresponding column labels are available as self.columns_, e.g.,
          ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

        """
        onehot = np.zeros((len(X), len(self.columns_)), dtype=int)
        for row_idx, transaction in enumerate(X):
            for col_idx, item in enumerate(self.columns_):
                onehot[row_idx, col_idx] = int(item in transaction)
        return onehot

    def inverse_transform(self, onehot):
        """Transforms a one-hot encoded NumPy array back into transactions.

        Parameters
        ------------
        onehot : NumPy array [n_transactions, n_unique_items]
           The NumPy one-hot encoded integer array of the input transactions,
           where the columns represent the unique items found in the input
           array in alphabetic order

           For example,
           ```
           array([[1, 0, 1, 1, 0, 1],
                  [1, 0, 1, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1],
                  [0, 0, 1, 0, 1, 1],
                  [0, 0, 1, 0, 1, 0],
                  [1, 1, 0, 0, 0, 0]])
          ```
          The corresponding column labels are available as self.columns_, e.g.,
          ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

        Returns
        ------------
        X : list of lists
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

        """
        return [[self.columns_[idx]
                 for idx, cell in enumerate(row) if cell]
                for row in onehot]

    def fit_transform(self, X):
        """Fit a OnehotTransactions encoder and transform a dataset."""
        return self.fit(X).transform(X)
