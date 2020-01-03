# Sebastian Raschka 2014-2019
# myxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from ..frequent_patterns import fpcommon as fpc


def generate_new_combinations(old_combinations):
    """
    Generator of all combinations based on the last state of Apriori algorithm
    Parameters
    -----------
    old_combinations: list of tuples
        All combinations with enough support in the last step
        Combinations are represented by a list of tuples.
        All tuples have the same length, which is equal to the combination size
        of the previous step.
        Each tuple represents one combination
        and contains item type ids in the ascending order
        ```
           15       20
           15       22
           17       19
        ```

    Returns
    -----------
    Generator of combinations based on the last state of Apriori algorithm.
    In order to reduce number of candidates, this function implements the
    apriori-gen function described in section 2.1.1 of Apriori paper.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/generate_new_combinations/

    """

    length = len(old_combinations)
    set_old_combinations = set(old_combinations)
    for i, old_combination in enumerate(old_combinations):
        head_i = list(old_combination[:-1])
        j = i + 1
        while j < length:
            *head_j, tail_j = old_combinations[j]
            if head_i != head_j:
                break
            # Prune old_combination+(item,) if any subset is not frequent
            candidate = old_combination + (tail_j,)
            # No need to check the last two values, because test_candidate
            # is then old_combinations[i] and old_combinations[j]
            for idx in range(len(candidate) - 2):
                test_candidate = list(candidate)
                del test_candidate[idx]
                if tuple(test_candidate) not in set_old_combinations:
                    # early exit from for-loop skips else clause just below
                    break
            else:
                yield candidate
            j = j + 1


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
             Apple  Bananas   Beer  Chicken   Milk   Rice
        0     True    False   True     True  False   True
        1     True    False   True    False  False   True
        2     True    False   True    False  False  False
        3     True     True  False    False  False  False
        4    False    False   True     True   True   True
        5    False    False   True    False   True   True
        6    False    False   True    False   True  False
        7     True     True  False    False  False  False
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
      Shows the number of combinations if >= 1.

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

    if min_support <= 0.:
        raise ValueError('`min_support` must be a positive '
                         'number within the interval `(0, 1]`. '
                         'Got %s.' % min_support)

    fpc.valid_input_check(df)

    # sparse attribute exists for both deprecated SparseDataFrame and
    # DataFrame with SparseArray (pandas >= 0.24); to_coo attribute
    # exists only for the former, thus it is checked first to distinguish
    # between SparseDataFrame and DataFrame with SparseArray.
    if hasattr(df, "to_coo"):
        # SparseDataFrame with pandas < 0.24
        if df.size == 0:
            X = df.values
        else:
            X = df.to_coo().tocsc()
            # See comment below
            X.eliminate_zeros()
        is_sparse = True
    elif hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            X = df.values
        else:
            X = df.sparse.to_coo().tocsc()
            # See comment below
            X.eliminate_zeros()
        is_sparse = True
    else:
        # dense DataFrame
        X = df.values
        is_sparse = False
    if is_sparse:
        # Count nonnull entries via direct access to X indices;
        # this requires X to be stored in CSC format, and to call
        # X.eliminate_zeros() to remove null entries from X.
        support = np.array([X.indptr[idx+1] - X.indptr[idx]
                            for idx in range(X.shape[1])], dtype=int)
    else:
        # Faster than np.count_nonzero(X, axis=0) or np.sum(X, axis=0), why?
        support = np.array([np.count_nonzero(X[:, idx])
                            for idx in range(X.shape[1])], dtype=int)
    support = support / X.shape[0]
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: [(idx,) for idx in np.where(support >= min_support)[0]]}
    max_itemset = 1

    while max_itemset and max_itemset < (max_len or float('inf')):
        next_max_itemset = max_itemset + 1

        combin = generate_new_combinations(itemset_dict[max_itemset])
        # count supports
        frequent_itemsets = []
        frequent_supports = []
        processed = 0
        if is_sparse:
            count = np.empty(X.shape[0], dtype=int)
            for itemset in combin:
                processed += 1
                count[:] = 0
                for item in itemset:
                    count[X.indices[X.indptr[item]:X.indptr[item+1]]] += 1
                support = np.count_nonzero(count == len(itemset)) / X.shape[0]
                if support >= min_support:
                    frequent_itemsets.append(itemset)
                    frequent_supports.append(support)
        else:
            _bools = np.empty(X.shape[0], dtype=bool)
            for itemset in combin:
                processed += 1
                _bools.fill(True)
                for item in itemset:
                    np.logical_and(_bools, X[:, item], out=_bools)
                support = np.count_nonzero(_bools) / X.shape[0]
                if support >= min_support:
                    frequent_itemsets.append(itemset)
                    frequent_supports.append(support)
        if not frequent_itemsets:
            # Exit condition
            break
        if verbose:
            print(
                '\rProcessed %d combinations | Sampling itemset size %d' %
                (processed, next_max_itemset), end="")
        itemset_dict[next_max_itemset] = frequent_itemsets
        support_dict[next_max_itemset] = frequent_supports
        max_itemset = next_max_itemset

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
