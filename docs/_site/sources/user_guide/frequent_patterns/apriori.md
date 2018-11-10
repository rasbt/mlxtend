## Frequent Itemsets via Apriori Algorithm

Apriori function to extract frequent itemsets for association rule mining

> from mlxtend.frequent_patterns import apriori

## Overview

Apriori is a popular algorithm [1] for extracting frequent itemsets with applications in association rule learning. The apriori algorithm has been designed to operate on databases containing transactions, such as purchases by customers of a store. An itemset is considered as "frequent" if it meets a user-specified support threshold. For instance, if the support threshold is set to 0.5 (50%), a frequent itemset is defined as a set of items that occur together in at least 50% of all transactions in the database.

## References

[1] Agrawal, Rakesh, and Ramakrishnan Srikant. "[Fast algorithms for mining association rules](https://www.it.uu.se/edu/course/homepage/infoutv/ht08/vldb94_rj.pdf)." Proc. 20th int. conf. very large data bases, VLDB. Vol. 1215. 1994.

## Example 1 -- Generating Frequent Itemsets

The `apriori` function expects data in a one-hot encoded pandas DataFrame.
Suppose we have the following transaction data:


```python
dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
```

We can transform it into the right format via the `TransactionEncoder` as follows:


```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Corn</th>
      <th>Dill</th>
      <th>Eggs</th>
      <th>Ice cream</th>
      <th>Kidney Beans</th>
      <th>Milk</th>
      <th>Nutmeg</th>
      <th>Onion</th>
      <th>Unicorn</th>
      <th>Yogurt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



Now, let us return the items and itemsets with at least 60% support:


```python
from mlxtend.frequent_patterns import apriori

apriori(df, min_support=0.6)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8</td>
      <td>(3)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>(5)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>(6)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>(8)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(10)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(3, 5)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(8, 3)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.6</td>
      <td>(5, 6)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(8, 5)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>(10, 5)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>(8, 3, 5)</td>
    </tr>
  </tbody>
</table>
</div>



By default, `apriori` returns the column indices of the items, which may be useful in downstream operations such as association rule mining. For better readability, we can set `use_colnames=True` to convert these integer values into the respective item names: 


```python
apriori(df, min_support=0.6, use_colnames=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8</td>
      <td>(Eggs)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>(Kidney Beans)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>(Milk)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>(Onion)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(Yogurt)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(Eggs, Kidney Beans)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Onion, Eggs)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.6</td>
      <td>(Milk, Kidney Beans)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(Onion, Kidney Beans)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>(Kidney Beans, Yogurt)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>(Onion, Eggs, Kidney Beans)</td>
    </tr>
  </tbody>
</table>
</div>



## Example 2 -- Selecting and Filtering Results

The advantage of working with pandas `DataFrames` is that we can use its convenient features to filter the results. For instance, let's assume we are only interested in itemsets of length 2 that have a support of at least 80 percent. First, we create the frequent itemsets via `apriori` and add a new column that stores the length of each itemset:


```python
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8</td>
      <td>(Eggs)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>(Kidney Beans)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>(Milk)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>(Onion)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(Yogurt)</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(Eggs, Kidney Beans)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Onion, Eggs)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.6</td>
      <td>(Milk, Kidney Beans)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(Onion, Kidney Beans)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>(Kidney Beans, Yogurt)</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>(Onion, Eggs, Kidney Beans)</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



Then, we can select the results that satisfy our desired criteria as follows:


```python
frequent_itemsets[ (frequent_itemsets['length'] == 2) &
                   (frequent_itemsets['support'] >= 0.8) ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(Eggs, Kidney Beans)</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, using the Pandas API, we can select entries based on the "itemsets" column:


```python
frequent_itemsets[ frequent_itemsets['itemsets'] == {'Onion', 'Eggs'} ]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Onion, Eggs)</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**Frozensets**

Note that the entries in the "itemsets" column are of type `frozenset`, which is built-in Python type that is similar to a Python `set` but immutable, which makes it more efficient for certain query or comparison operations (https://docs.python.org/3.6/library/stdtypes.html#frozenset). Since `frozenset`s are sets, the item order does not matter. I.e., the query

`frequent_itemsets[ frequent_itemsets['itemsets'] == {'Onion', 'Eggs'} ]`
    
is equivalent to any of the following three

- `frequent_itemsets[ frequent_itemsets['itemsets'] == {'Eggs', 'Onion'} ]`
- `frequent_itemsets[ frequent_itemsets['itemsets'] == frozenset(('Eggs', 'Onion')) ]`
- `frequent_itemsets[ frequent_itemsets['itemsets'] == frozenset(('Onion', 'Eggs')) ]`




## Example 3 -- Working with Sparse Representations

To save memory, you may want to represent your transaction data in the sparse format.
This is especially useful if you have lots of products and small transactions.


```python
oht_ary = te.fit(dataset).transform(dataset, sparse=True)
sparse_df = pd.SparseDataFrame(te_ary, columns=te.columns_, default_fill_value=False)
sparse_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Apple</th>
      <th>Corn</th>
      <th>Dill</th>
      <th>Eggs</th>
      <th>Ice cream</th>
      <th>Kidney Beans</th>
      <th>Milk</th>
      <th>Nutmeg</th>
      <th>Onion</th>
      <th>Unicorn</th>
      <th>Yogurt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>True</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>




```python
apriori(sparse_df, min_support=0.6, use_colnames=True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>support</th>
      <th>itemsets</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.8</td>
      <td>(Eggs)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>(Kidney Beans)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6</td>
      <td>(Milk)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>(Onion)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.6</td>
      <td>(Yogurt)</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.8</td>
      <td>(Eggs, Kidney Beans)</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.6</td>
      <td>(Onion, Eggs)</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.6</td>
      <td>(Milk, Kidney Beans)</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.6</td>
      <td>(Onion, Kidney Beans)</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.6</td>
      <td>(Kidney Beans, Yogurt)</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.6</td>
      <td>(Onion, Eggs, Kidney Beans)</td>
    </tr>
  </tbody>
</table>
</div>



## API


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


