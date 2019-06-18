# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import collections
import math
import itertools
import numpy as np
import pandas as pd
import mlxtend.frequent_patterns.fpcommon as fpc


def fpgrowth(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
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

    """
    fpc.valid_input_check(df)

    itemsets = df.values
    num_items = itemsets.shape[1]       # number of unique items
    num_itemsets = itemsets.shape[0]    # number of itemsets in the database
    # support of each individual item
    item_support = np.sum(itemsets, axis=0) / float(num_itemsets)

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    items = [item for item in range(
        num_items) if item_support[item] >= min_support]

    # Define ordering on items for inserting into FPTree
    items.sort(key=lambda x: item_support[x])
    rank = {item: i for i, item in enumerate(items)}

    # Building tree by inserting itemsets in sorted order
    # Hueristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = fpc.FPTree(rank)
    for i in range(num_itemsets):
        itemset = [item for item in np.where(
            itemsets[i, :])[0] if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    # Collect frequent itemsets
    minsup = math.ceil(min_support * num_itemsets)  # min support as count
    frequent_itemsets = []
    itemset_supports = []

    for sup, iset in fpg_step(tree, minsup, colname_map, max_len, verbose):
        frequent_itemsets.append(frozenset(iset))
        itemset_supports.append(sup/num_itemsets)

    res_df = pd.DataFrame({'support': itemset_supports,
                           'itemsets': frequent_itemsets})

    if use_colnames:
        res_df['itemsets'] = res_df['itemsets'] \
            .apply(lambda x: frozenset([colname_map[i] for i in x]))

    return res_df


def fpg_step(tree, minsup, colnames, max_len, verbose):
    """
    Performs a recursive step of the fpgrowth algorithm.

    Parameters
    ----------
    tree : FPTree
    minsup : int

    Yields
    ------
    lists of strings
        Set of items that has occurred in minsup itemsets.
    """
    count = 0
    items = tree.nodes.keys()
    if tree.is_path():
        # If the tree is a path, we can combinatorally generate all
        # remaining itemsets without generating additional conditional trees
        size_remain = len(items) + 1
        if max_len:
            size_remain = max_len - len(tree.cond_items) + 1
        for i in range(1, size_remain):
            for itemset in itertools.combinations(items, i):
                count += 1
                support = min([tree.nodes[i][0].count for i in itemset])
                yield support, tree.cond_items + list(itemset)
    elif not max_len or max_len > len(tree.cond_items):
        for item in items:
            count += 1
            support = sum([node.count for node in tree.nodes[item]])
            yield support, tree.cond_items + [item]

    if verbose:
        tree.print_status(count, colnames)

    # Generate conditional trees to generate frequent itemsets one item larger
    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        for item in items:
            cond_tree = tree.conditional_tree(item, minsup)
            for sup, iset in fpg_step(cond_tree, minsup,
                                      colnames, max_len, verbose):
                yield sup, iset