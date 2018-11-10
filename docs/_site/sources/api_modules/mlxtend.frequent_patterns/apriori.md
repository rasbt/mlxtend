## apriori

*apriori(df, min_support=0.5, use_colnames=False, max_len=None, n_jobs=1)*

Get frequent itemsets from a one-hot DataFrame
**Parameters**

- `df` : pandas DataFrame or pandas SparseDataFrame

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


- `min_support` : float (default: 0.5)

    A float between 0 and 1 for minumum support of the itemsets returned.
    The support is computed as the fraction
    transactions_where_item(s)_occur / total_transactions.


- `use_colnames` : bool (default: False)

    If true, uses the DataFrames' column names in the returned DataFrame
    instead of column indices.


- `max_len` : int (default: None)

    Maximum length of the itemsets generated. If `None` (default) all
    possible itemsets lengths (under the apriori condition) are evaluated.

**Returns**

pandas DataFrame with columns ['support', 'itemsets'] of all itemsets
    that are >= `min_support` and < than `max_len`
    (if `max_len` is not None).
    Each itemset in the 'itemsets' column is of type `frozenset`,
    which is a Python built-in type that behaves similarly to
    sets except that it is immutable
    (For more info, see
    https://docs.python.org/3.6/library/stdtypes.html#frozenset).

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/apriori/)

