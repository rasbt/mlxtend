# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
import numpy as np
import pandas as pd


def apriori(df, min_support=0.5, use_colnames=False):
    """Get frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame in one-hot encoded format. For example
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
      transactions_where_item(s)_occur / total_transactions.
    use_colnames : bool (default: False)
      If true, uses the DataFrames' column names in the returned DataFrame
      instead of column indices.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
    that are >= min_support.

    """

    X = df.values
    ary_col_idx = np.arange(X.shape[1])
    support = (np.sum(X, axis=0) / float(X.shape[0]))
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1

    while max_itemset:
        next_max_itemset = max_itemset + 1
        combin = combinations(np.unique(itemset_dict[max_itemset].flatten()),
                              r=next_max_itemset)
        frequent_items = []
        frequent_items_support = []

        for c in combin:
            together = X[:, c].sum(axis=1) == len(c)
            support = together.sum() / float(X.shape[0])
            if support >= min_support:
                frequent_items.append(c)
                frequent_items_support.append(support)

        if frequent_items:
            itemset_dict[next_max_itemset] = np.array(frequent_items)
            support_dict[next_max_itemset] = np.array(frequent_items_support)
            max_itemset = next_max_itemset
        else:
            max_itemset = 0

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([i for i in itemset_dict[k]])

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ['support', 'itemsets']
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df['itemsets'] = res_df['itemsets'].apply(lambda x: [mapping[i]
                                                      for i in x])
    res_df['length'] = res_df['itemsets'].apply(lambda x: len(x))

    return res_df
