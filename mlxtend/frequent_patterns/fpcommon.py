import numpy as np
import pandas as pd
import collections


def setup_fptree(df, min_support):
    itemsets = df.values
    num_items = itemsets.shape[1]       # number of unique items
    num_itemsets = itemsets.shape[0]    # number of itemsets in the database

    # support of each individual item
    item_support = np.sum(itemsets, axis=0) / float(num_itemsets)

    items = [item for item in range(
        num_items) if item_support[item] >= min_support]

    # Define ordering on items for inserting into FPTree
    items.sort(key=lambda x: item_support[x])
    rank = {item: i for i, item in enumerate(items)}

    # Building tree by inserting itemsets in sorted order
    # Hueristic for reducing tree size is inserting in order
    #   of most frequent to least frequent
    tree = FPTree(rank)
    for i in range(num_itemsets):
        itemset = [item for item in np.where(
            itemsets[i, :])[0] if item in rank]
        itemset.sort(key=rank.get, reverse=True)
        tree.insert_itemset(itemset)

    return tree, rank


def generate_itemsets(generator, num_itemsets, colname_map):
    itemsets = []
    supports = []
    for sup, iset in generator:
        itemsets.append(frozenset(iset))
        supports.append(sup / num_itemsets)

    res_df = pd.DataFrame({'support': supports, 'itemsets': itemsets})

    if colname_map is not None:
        res_df['itemsets'] = res_df['itemsets'] \
            .apply(lambda x: frozenset([colname_map[i] for i in x]))

    return res_df


def valid_input_check(df):
    # Pandas is much slower than numpy, so use df.values instead of df here
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


class FPTree(object):
    def __init__(self, rank=None):
        self.root = FPNode(None)
        self.nodes = collections.defaultdict(list)
        self.cond_items = []
        self.rank = rank

    def conditional_tree(self, cond_item, minsup):
        """
        Creates and returns the subtree of self conditioned on cond_item.

        Parameters
        ----------
        cond_item : int | str
            Item that the tree (self) will be conditioned on.
        minsup : int
            Minimum support threshold.

        Returns
        -------
        cond_tree : FPtree
        """
        # Find all path from root node to nodes for item
        branches = []
        count = collections.defaultdict(int)
        for node in self.nodes[cond_item]:
            branch = node.itempath_from_root()
            branches.append(branch)
            for item in branch:
                count[item] += node.count

        # Define new ordering or deep trees may have combinatorially explosion
        items = [item for item in count if count[item] >= minsup]
        items.sort(key=count.get)
        rank = {item: i for i, item in enumerate(items)}

        # Create conditional tree
        cond_tree = FPTree(rank)
        for idx, branch in enumerate(branches):
            branch = sorted([i for i in branch if i in rank],
                            key=rank.get, reverse=True)
            cond_tree.insert_itemset(branch, self.nodes[cond_item][idx].count)
        cond_tree.cond_items = self.cond_items + [cond_item]

        return cond_tree

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
                child.count += count
                node = child
                index += 1
            else:
                break

        # Insert any remaining items
        for item in itemset[index:]:
            child_node = FPNode(item, count, node)
            self.nodes[item].append(child_node)
            node = child_node

    def is_path(self):
        if len(self.root.children) > 1:
            return False
        for i in self.nodes:
            if len(self.nodes[i]) > 1 or len(self.nodes[i][0].children) > 1:
                return False
        return True

    def print_status(self, count, colnames):
        cond_items = [str(i) for i in self.cond_items]
        if colnames:
            cond_items = [str(colnames[i]) for i in self.cond_items]
        cond_items = ", ".join(cond_items)
        print('\r%d itemset(s) from tree conditioned on items (%s)' %
              (count, cond_items), end="\n")


class FPNode(object):
    def __init__(self, item, count=1, parent=None):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = collections.defaultdict(FPNode)

        if parent is not None:
            parent.children[item] = self

    def itempath_from_root(self):
        """ Returns the top-down sequence of items from self to
            (but not including) the root node. """
        path = []
        if self.item is None:
            return path

        node = self.parent
        while node.item is not None:
            path.append(node.item)
            node = node.parent

        path.reverse()
        return path
