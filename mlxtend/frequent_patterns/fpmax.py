# mlxtend Machine Learning Library Extensions
# Author: Steve Harenberg <harenbergsd@gmail.com>
#
# License: BSD 3 clause

import collections
import math

from ..frequent_patterns import fpcommon as fpc


def fpmax(df, min_support=0.5, use_colnames=False, max_len=None, verbose=0):
    """Get maximal frequent itemsets from a one-hot DataFrame

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame the encoded format. Also supports
      DataFrames with sparse data; for more info, please
      see (https://pandas.pydata.org/pandas-docs/stable/
           user_guide/sparse.html#sparse-data-structures)

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
      Given the set of all maximal itemsets,
      return those that are less than `max_len`. If `None` (default) all
      possible itemsets lengths are evaluated.

    verbose : int (default: 0)
      Shows the stages of conditional tree generation.

    Returns
    -----------
    pandas DataFrame with columns ['support', 'itemsets'] of all maximal
      itemsets that are >= `min_support` and < than `max_len`
      (if `max_len` is not None).
      Each itemset in the 'itemsets' column is of type `frozenset`,
      which is a Python built-in type that behaves similarly to
      sets except that it is immutable
      (For more info, see
      https://docs.python.org/3.6/library/stdtypes.html#frozenset).

    Examples
    ----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpmax/

    """
    fpc.valid_input_check(df)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )

    colname_map = None
    if use_colnames:
        colname_map = {idx: item for idx, item in enumerate(df.columns)}

    tree, rank = fpc.setup_fptree(df, min_support)

    minsup = math.ceil(min_support * len(df))  # min support as count
    generator = fpmax_step(tree, minsup, MFITree(rank), colname_map, max_len, verbose)

    return fpc.generate_itemsets(generator, len(df), colname_map)


def fpmax_step(tree, minsup, mfit, colnames, max_len, verbose):
    count = 0
    items = list(tree.nodes.keys())
    largest_set = sorted(tree.cond_items + items, key=mfit.rank.get)
    if len(largest_set) == 0:
        return
    if tree.is_path():
        if not mfit.contains(largest_set):
            count += 1
            largest_set.reverse()
            mfit.cache = largest_set
            mfit.insert_itemset(largest_set)
            if max_len is None or len(largest_set) <= max_len:
                support = tree.root.count
                if len(items) > 0:
                    support = min([tree.nodes[i][0].count for i in items])
                yield support, largest_set

    if verbose:
        tree.print_status(count, colnames)

    if not tree.is_path() and (not max_len or max_len > len(tree.cond_items)):
        # Loop over each item in tree creating another conditional tree
        items.sort(key=tree.rank.get)
        for item in items:
            # Check if the tree will produce a subset already produced
            if mfit.contains(largest_set):
                return
            largest_set.remove(item)
            cond_tree = tree.conditional_tree(item, minsup)
            for support, mfi in fpmax_step(
                cond_tree, minsup, mfit, colnames, max_len, verbose
            ):
                yield support, mfi


class MFITree(object):
    def __init__(self, rank):
        self.root = self.Node(None)
        self.nodes = collections.defaultdict(list)
        self.cache = []
        self.rank = rank

    def insert_itemset(self, itemset, count=1):
        """
        Inserts a list of items into the tree.

        Parameters
        ----------
        itemset : list
            Items that will be inserted into the tree.
        count : int
            The number of occurrences of the itemset.
        """
        if len(itemset) == 0:
            return

        # Follow existing path in tree as long as possible
        index = 0
        node = self.root
        for item in itemset:
            if item in node.children:
                child = node.children[item]
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = self.Node(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def contains(self, itemset):
        """
        Checks if this tree contains itemset in one of its branches.
        The algorithm assumes that  itemset is sorted according to self.rank.
        """
        i = 0
        for item in reversed(self.cache):
            if self.rank[itemset[i]] < self.rank[item]:
                break
            if itemset[i] == item:
                i += 1
            if i == len(itemset):
                return True

        for basenode in self.nodes[itemset[0]]:
            i = 0
            node = basenode
            while node.item is not None:
                if self.rank[itemset[i]] < self.rank[node.item]:
                    break
                if itemset[i] == node.item:
                    i += 1
                if i == len(itemset):
                    return True
                node = node.parent

        return False

    class Node(object):
        def __init__(self, item, count=1, parent=None):
            self.item = item
            self.parent = parent
            self.children = collections.defaultdict(MFITree.Node)

            if parent is not None:
                parent.children[item] = self
