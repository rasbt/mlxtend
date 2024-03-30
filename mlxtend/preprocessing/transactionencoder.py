# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import _check_feature_names_in, check_is_fitted


class TransactionEncoder(BaseEstimator, TransformerMixin):
    """Encoder class for transaction data in Python lists

    Parameters
    ------------
    None

    Attributes
    ------------
    columns_: list
      List of unique names in the `X` input list of lists

    Examples
    ------------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/

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
        columns_mapping = {}
        for col_idx, item in enumerate(self.columns_):
            columns_mapping[item] = col_idx
        self.columns_mapping_ = columns_mapping
        return self

    def transform(self, X, sparse=False):
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

        sparse: bool (default=False)
          If True, transform will return Compressed Sparse Row matrix
          instead of the regular one.

        Returns
        ------------
        array : NumPy array [n_transactions, n_unique_items]
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
        """
        if sparse:
            indptr = [0]
            indices = []
            for transaction in X:
                # set is necessary because conversion to SparseDataFrame
                # will fail if there are duplicate items
                for item in set(transaction):
                    col_idx = self.columns_mapping_[item]
                    indices.append(col_idx)
                indptr.append(len(indices))
            non_sparse_values = [True] * len(indices)
            array = csr_matrix((non_sparse_values, indices, indptr), dtype=bool)
        else:
            array = np.zeros((len(X), len(self.columns_)), dtype=bool)
            for row_idx, transaction in enumerate(X):
                for item in transaction:
                    col_idx = self.columns_mapping_[item]
                    array[row_idx, col_idx] = True
        return array

    def inverse_transform(self, array):
        """Transforms an encoded NumPy array back into transactions.

        Parameters
        ------------
        array : NumPy array [n_transactions, n_unique_items]
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
        return [
            [self.columns_[idx] for idx, cell in enumerate(row) if cell]
            for row in array
        ]

    def fit_transform(self, X, sparse=False):
        """Fit a TransactionEncoder encoder and transform a dataset."""
        return self.fit(X).transform(X, sparse=sparse)

    def get_feature_names_out(self):
        """Used to get the column names of pandas output.

        This method combined with the `TransformerMixin` exposes the
        set_output API to the `TransactionEncoder`. This allows the user
        to set the transformed output to a `pandas.DataFrame` by default.

        See  https://scikit-learn.org/stable/developers/develop.html#developer-api-set-output
        for more details.
        """
        check_is_fitted(self, attributes="columns_")
        return _check_feature_names_in(estimator=self, input_features=self.columns_)
