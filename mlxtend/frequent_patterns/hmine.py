# mlxtend Machine Learning Library Extensions
# Author: Fatih Sen <fatih.sn2000@gmail.com>
#
# License: BSD 3 clause

import math

import numpy as np
import pandas as pd

from ..frequent_patterns import fpcommon as fpc


def hmine(
    df, min_support=0.5, use_colnames=False, max_len=None, verbose=0
) -> pd.DataFrame:
    """
    Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#sparse-data-structures.

      Please note that the old pandas SparseDataFrame format
      is no longer supported in mlxtend >= 0.17.2.

      The allowed values are either 0/1 or True/False.
      For example,

    ```
           Apple  Bananas   Beer  Chicken   Milk   Rice
        0   True    False   True     True  False   True
        1   True    False   True    False  False   True
        2   True    False   True    False  False  False
        3   True     True  False    False  False  False
        4  False    False   True     True   True   True
        5  False    False   True    False   True   True
        6  False    False   True    False   True  False
        7   True     True  False    False  False  False
    ```

    min_support : float (default: 0.5)
      A float between 0 and 1 for minimum support of the itemsets returned.
      The support is computed as the fraction
      transactions_where_item(s)_occur / total_transactions.

    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    max_len : int (default: None)
      Maximum length of the itemsets generated. If `None` (default) all
      possible itemsets lengths are evaluated.

    verbose : int (default: 0)
      Shows the stages of conditional tree generation.

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
    ----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/

    """

    fpc.valid_input_check(df)
    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    # Calculate the minimum support based on the number of transactions (absolute support)
    minsup = math.ceil(min_support * len(df))

    is_sparse = False
    if hasattr(df, "sparse"):
        # DataFrame with SparseArray (pandas >= 0.24)
        if df.size == 0:
            itemsets = df.values
        else:
            itemsets = df.sparse.to_coo().tocsr()
            is_sparse = True
    else:
        # dense DataFrame
        itemsets = df.values
    if is_sparse:
        is_sparse
    single_items = np.array(df.columns)
    itemsets_shape = itemsets.shape[0]
    itemsets, single_items, single_items_support = itemset_optimisation(
        itemsets, single_items, minsup
    )
    numeric_single_items = np.arange(len(single_items))
    frequent_itemsets = {}
    for item in numeric_single_items:
        if single_items_support[item] >= minsup:
            supp = single_items_support[item] / itemsets_shape
            frequent_itemsets[frozenset([single_items[item]])] = supp
        if max_len == 1:
            continue
        # Recursive call to find frequent itemsets
        frequent_itemsets = hmine_driver(
            [item],
            itemsets,
            minsup,
            itemsets_shape,
            max_len,
            verbose,
            single_items,
            frequent_itemsets,
        )
    # Convert the dictionary to a DataFrame
    res_df = pd.DataFrame([frequent_itemsets.values(), frequent_itemsets.keys()]).T
    res_df.columns = ["support", "itemsets"]

    if not use_colnames:
        mapping = {item: idx for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )

    return res_df


def itemset_optimisation(
    itemsets: np.array,
    single_items: np.array,
    minsup: int,
) -> tuple:
    """
    Downward-closure property of H-Mine algorithm.
        Optimizes the itemsets matrix by removing items that do not
        meet the minimum support. (For more info, see
        https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/)

    Args:
        itemsets (np.array): matrix of bools or binary
        single_items (np.array): array of single items
        minsup (int): minimum absolute support

    Returns:
        itemsets (np.array): reduced itemsets matrix of bools or binary
        single_items (np.array): reduced array of single items
        single_items_support (np.array): reduced single items support
    """

    single_items_support = np.array(np.sum(itemsets, axis=0)).reshape(-1)
    items = np.nonzero(single_items_support >= minsup)[0]
    itemsets = itemsets[:, items]
    single_items = single_items[items]
    single_items_support = single_items_support[items]

    return itemsets, single_items, single_items_support


def hmine_driver(
    item: list,
    itemsets: np.array,
    minsup: int,
    itemsets_shape: int,
    max_len: int,
    verbose: int,
    single_items: np.array,
    frequent_itemsets: dict,
) -> dict:
    """
    Driver function for the hmine algorithm.
    Recursively generates frequent itemsets.
    Also works for sparse matrix.
    egg: item = [1] -> [1,2] -> [1,2,3] -> [1,2,4] -> [1,2,5]
    For more info, see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/

    Args:
        item (list): list of items
        itemsets (np.array): matrix of bools or binary
        minsup (int): minimum absolute support
        itemsets_shape (int): number of transactions
        single_items (np.array): array of single items
        max_len (int): maximum length of frequent itemsets
        verbose (int): verbose mode
        frequent_itemsets (dict): dictionary of frequent itemsets

    Returns:
        frequent_itemsets(dict): dictionary of frequent itemsets
    """
    # Early stopping if the length of the item is greater than max_len
    if max_len and len(item) >= max_len:
        return frequent_itemsets
    projected_itemsets = create_projected_itemsets(item, itemsets)
    initial_supports = np.array(np.sum(projected_itemsets, axis=0)).reshape(-1)
    suffixes = np.nonzero(initial_supports >= minsup)[0]
    suffixes = suffixes[np.nonzero(suffixes > item[-1])[0]]

    if verbose:
        print(
            f"{len(suffixes)} itemset(s) from the suffixes on item(s) ({', '.join(single_items[item])})"
        )
    for suffix in suffixes:
        new_item = item.copy()
        new_item.append(suffix)
        supp = initial_supports[suffix] / itemsets_shape
        frequent_itemsets[frozenset(single_items[new_item])] = supp
        # Recursive call to find frequent itemsets
        frequent_itemsets = hmine_driver(
            new_item,
            projected_itemsets,
            minsup,
            itemsets_shape,
            max_len,
            verbose,
            single_items,
            frequent_itemsets,
        )
    return frequent_itemsets


def create_projected_itemsets(item: list, itemsets: np.array) -> np.array:
    """
    Creates the projected itemsets for the given item. (For more info, see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/)

    Args:
        item (list): list of items
        itemsets (np.array): matrix of bools or binary

    Returns:
        projected_itemsets(np.array): projected itemsets for the given item
    """

    indices = np.nonzero(np.sum(itemsets[:, item], axis=1) == len(item))[0]
    projected_itemsets = itemsets[indices, :]
    return projected_itemsets
