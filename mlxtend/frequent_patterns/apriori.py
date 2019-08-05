# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd


def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: np.array
        All combinations with enough support in the last step
        Combinations are represented by a matrix.
        Number of columns is equal to the combination size
        of the previous step.
        Each row represents one combination
        and contains item type ids in the ascending order
        ```
               0        1
        0      15       20
        1      15       22
        2      17       19
        ```

    Returns
    -----------
    Generator of all combinations from the last step x items
    from the previous step. Every combination is a tuple
    of item type ids in the ascending order.
    No combination other than generated
    do not have a chance to get enough support

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = max(old_combination)
        for item in items_types_in_previous_step:
            if item > max_combination:
                res = tuple(old_combination) + (item,)
                yield res


def apriori(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0,
            low_memory=False):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame or pandas SparseDataFrame
      pandas DataFrame the encoded format.
      The allowed values are either 0/1 or True/False.
      For example,

    ```
             Apple  Bananas  Beer  Chicken  Milk  Rice
        0      1        0     1        1     0     1
        1      1        0     1        0     0     1
        2      1        0     1        0     0     0
        3      1        1     0        0     0     0
        4      0        0     1        1     1     1
        5      0        0     1        0     1     1
        6      0        0     1        0     1     0
        7      1        1     0        0     0     0
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minumum support of the itemsets returned.
      The support is computed as the fraction
      `transactions_where_item(s)_occur / total_transactions`.

    use_colnames : bool (default: False)
      If `True`, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths (under the apriori condition) are evaluated.

    verbose : int (default: 0)
      Shows the number of iterations if >= 1 and `low_memory` is `True`. If
      >=1 and `low_memory` is `False`, shows the number of combinations.

    low_memory : bool (default: False)
      If `True`, uses an iterator to search for combinations above
      `min_support`.
      Note that while `low_memory=True` should only be used for large dataset
      if memory resources are limited, because this implementation is approx.
      3-6x slower than the default.


    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
      that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

    """

    def _support(_x, _n_rows, _is_sparse):
        """DRY private method to calculate support as the
        row-wise sum of values / number of rows

        Parameters
        -----------

        _x : matrix of bools or binary

        _n_rows : numeric, number of rows in _x

        _is_sparse : bool True if _x is sparse

        Returns
        -----------
        np.array, shape = (n_rows, )

        Examples
        -----------
        For usage examples, please see
        http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/

        """
        out = (np.sum(_x, axis=0) / _n_rows)
        return np.array(out).reshape(-1)

    idxs = np.where((df.values != 1) & (df.values != 0))
    if len(idxs[0]) > 0:
        val = df.values[idxs[0][0], idxs[1][0]]
        s = ('The allowed values for a DataFrame'
             ' are True, False, 0, 1. Found value %s' % (val))
        raise ValueError(s)

    is_sparse = hasattr(df, "to_coo")
    if is_sparse:
        if not isinstance(df.columns[0], str) and df.columns[0] != 0:
            raise ValueError('Due to current limitations in Pandas, '
                             'if the SparseDataFrame has integer column names,'
                             'names, please make sure they either start '
                             'with `0` or cast them as string column names: '
                             '`df.columns = [str(i) for i in df.columns`].')
        X = df.to_coo().tocsc()
    else:
        X = df.values
    support = _support(X, X.shape[0], is_sparse)
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    iter_count = 0
    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float('inf')):
        next_max_itemset = max_itemset + 1
        combin = generate_new_combinations(itemset_dict[max_itemset])

        # With exceptionally large datasets, the matrix operations can use a
        # substantial amount of memory. For low memory applications or large
        # datasets, set `low_memory=True` to use a slower but more memory-
        # efficient implementation.
        if low_memory:
            frequent_items = []
            frequent_items_support = []
            if is_sparse:
                all_ones = np.ones((X.shape[0], next_max_itemset))
            for c in combin:
                if verbose:
                    iter_count += 1
                    print('\rIteration: %d | Sampling itemset size %d' %
                          (iter_count, next_max_itemset), end="")
                if is_sparse:
                    together = np.all(X[:, c] == all_ones, axis=1)
                else:
                    together = X[:, c].all(axis=1)
                support = together.sum() / rows_count
                if support >= min_support:
                    frequent_items.append(c)
                    frequent_items_support.append(support)

            if frequent_items:
                itemset_dict[next_max_itemset] = np.array(frequent_items)
                support_dict[next_max_itemset] = \
                    np.array(frequent_items_support)
                max_itemset = next_max_itemset
            else:
                break
        else:
            combin = np.array(list(combin))

            if combin.size == 0:
                break
            if verbose:
                print(
                    '\rProcessing %d combinations | Sampling itemset size %d' %
                    (combin.size, next_max_itemset), end="")

            if is_sparse:
                _bools = X[:, combin[:, 0]] == all_ones
                for n in range(1, combin.shape[1]):
                    _bools = _bools & (X[:, combin[:, n]] == all_ones)
            else:
                _bools = np.all(X[:, combin], axis=2)

            support = _support(np.array(_bools), rows_count, is_sparse)
            _mask = (support >= min_support).reshape(-1)
            if any(_mask):
                itemset_dict[next_max_itemset] = np.array(combin[_mask])
                support_dict[next_max_itemset] = np.array(support[_mask])
                max_itemset = next_max_itemset
            else:
                # Exit condition
                break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: frozenset([
                                                      mapping[i] for i in x]))
    res_df = res_df.reset_index(drop=True)

    if verbose:
        print()  # adds newline if verbose counter was used

    return res_df
