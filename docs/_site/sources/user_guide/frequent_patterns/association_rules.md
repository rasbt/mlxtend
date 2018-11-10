## Association Rules Generation from Frequent Itemsets

Function to generate association rules from frequent itemsets

> from mlxtend.frequent_patterns import association_rules

## Overview

Rule generation is a common task in the mining of frequent patterns. _An association rule is an implication expression of the form $X \rightarrow Y$, where $X$ and $Y$ are disjoint itemsets_ [1]. A more concrete example based on consumer behaviour would be  $\{Diapers\} \rightarrow \{Beer\}$ suggesting that people who buy diapers are also likely to buy beer. To evaluate the "interest" of such an association rule, different metrics have been developed. The current implementation make use of the `confidence` and `lift` metrics. 


### Metrics

The currently supported metrics for evaluating association rules and setting selection thresholds are listed below. Given a rule "A -> C", *A* stands for antecedent and *C* stands for consequent.


#### 'support':

$$\text{support}(A\rightarrow C) = \text{support}(A \cup C), \;\;\; \text{range: } [0, 1]$$

- introduced in [3]

The support metric is defined for itemsets, not assocication rules. The table produced by the association rule mining algorithm contains three different support metrics: 'antecedent support', 'consequent support', and 'support'. Here, 'antecedent support' computes the proportion of transactions that contain the antecedent A, and 'consequent support' computes the support for the itemset of the consequent C. The 'support' metric then computes the support of the combined itemset A $\cup$ C -- note that 'support' depends on 'antecedent support' and 'consequent support' via min('antecedent support', 'consequent support').


Typically, support is used to measure the abundance or frequency (often interpreted as significance or importance) of an itemset in a database. We refer to an itemset as a "frequent itemset" if you support is larger than a specified minimum-support threshold. Note that in general, due to the *downward closure* property, all subsets of a frequent itemset are also frequent.


#### 'confidence':  

$$\text{confidence}(A\rightarrow C) = \frac{\text{support}(A\rightarrow C)}{\text{support}(A)}, \;\;\; \text{range: } [0, 1]$$

- introduced in [3]

The confidence of a rule A->C is the probability of seeing the consequent in a transaction given that it also contains the antecedent. Note that the metric is not symmetric or directed; for instance, the confidence for A->C is different than the confidence for C->A. The confidence is 1 (maximal) for a rule A->C if the consequent and antecedent always occur together. 


#### 'lift':

$$\text{lift}(A\rightarrow C) = \frac{\text{confidence}(A\rightarrow C)}{\text{support}(C)}, \;\;\; \text{range: } [0, \infty]$$


- introduced in [4]


The lift metric is commonly used to measure how much more often the antecedent and consequent of a rule A->C occur together than we would expect if they were statistically independent. If A and C are independent, the Lift score will be exactly 1.


#### 'leverage':

$$\text{levarage}(A\rightarrow C) = \text{support}(A\rightarrow C) - \text{support}(A) \times \text{support}(C), \;\;\; \text{range: } [-1, 1]$$


- introduced in [5]

Leverage computes the difference between the observed frequency of A and C appearing together and the frequency that would be expected if A and C were independent. An leverage value of 0 indicates independence.

#### 'conviction':

$$\text{conviction}(A\rightarrow C) = \frac{1 - \text{support}(C)}{1 - \text{confidence}(A\rightarrow C)}, \;\;\; \text{range: } [0, \infty]$$

- introduced in [6]

A high conviction value means that the consequent is highly depending on the antecedent. For instance, in the case of a perfect confidence score, the denominator becomes 0 (due to 1 - 1) for which the conviction score is defined as 'inf'. Similar to lift, if items are independent, the conviction is 1.
 

## References


[1] Tan, Steinbach, Kumar. Introduction to Data Mining. Pearson New International Edition. Harlow: Pearson Education Ltd., 2014. (pp. 327-414).

[2] Michael Hahsler, http://michael.hahsler.net/research/association_rules/measures.html

[3] R. Agrawal, T. Imielinski, and A. Swami. Mining associations between sets of items in large databases. In Proc. of the ACM SIGMOD Int'l Conference on Management of Data, pages 207-216, Washington D.C., May 1993

[4] S. Brin, R. Motwani, J. D. Ullman, and S. Tsur. Dynamic itemset counting and implication rules for market basket data

[5]  Piatetsky-Shapiro, G., Discovery, analysis, and presentation of strong rules. Knowledge Discovery in Databases, 1991: p. 229-248.

[6] Sergey Brin, Rajeev Motwani, Jeffrey D. Ullman, and Shalom Turk. Dynamic itemset counting and implication rules for market basket data. In SIGMOD 1997, Proceedings ACM SIGMOD International Conference on Management of Data, pages 255-264, Tucson, Arizona, USA, May 1997

## Example 1 -- Generating Association Rules from Frequent Itemsets

The `generate_rules` takes dataframes of frequent itemsets as produced by the `apriori` function in *mlxtend.association*. To demonstrate the usage of the `generate_rules` method, we first create a pandas `DataFrame` of frequent itemsets as generated by the [`apriori`](./apriori.md) function:



```python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori


dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

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
      <td>(Kidney Beans, Eggs)</td>
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
      <td>(Onion, Kidney Beans, Eggs)</td>
    </tr>
  </tbody>
</table>
</div>



The `generate_rules()` function allows you to (1) specify your metric of interest and (2) the according threshold. Currently implemented measures are **confidence** and **lift**. Let's say you are interesting in rules derived from the frequent itemsets only if the level of confidence is above the 90 percent threshold (`min_threshold=0.7`):


```python
from mlxtend.frequent_patterns import association_rules

association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>0.8</td>
      <td>0.80</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Eggs)</td>
      <td>(Kidney Beans)</td>
      <td>0.8</td>
      <td>1.0</td>
      <td>0.8</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Onion)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Milk)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Onion)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(Yogurt)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(Onion, Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(Onion, Eggs)</td>
      <td>(Kidney Beans)</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(Kidney Beans, Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(Onion)</td>
      <td>(Kidney Beans, Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(Eggs)</td>
      <td>(Onion, Kidney Beans)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
  </tbody>
</table>
</div>



## Example 2 -- Rule Generation and Selection Criteria

If you are interested in rules according to a different metric of interest, you can simply adjust the `metric` and `min_threshold` arguments . E.g. if you are only interested in rules that have a lift score of >= 1.2, you would do the following:


```python
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Onion)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Onion, Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Kidney Beans, Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Onion)</td>
      <td>(Kidney Beans, Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Eggs)</td>
      <td>(Onion, Kidney Beans)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
    </tr>
  </tbody>
</table>
</div>



Pandas `DataFrames` make it easy to filter the results further. Let's say we are ony interested in rules that satisfy the following criteria:

1. at least 2 antecedents
2. a confidence > 0.75
3. a lift score > 1.2

We could compute the antecedent length as follows:


```python
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>antecedent_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(Onion)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(Onion, Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(Kidney Beans, Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(Onion)</td>
      <td>(Kidney Beans, Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.00</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(Eggs)</td>
      <td>(Onion, Kidney Beans)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.600000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Then, we can use pandas' selection syntax as shown below:


```python
rules[ (rules['antecedent_len'] >= 2) &
       (rules['confidence'] > 0.75) &
       (rules['lift'] > 1.2) ]
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>antecedent_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>(Onion, Kidney Beans)</td>
      <td>(Eggs)</td>
      <td>0.6</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>1.0</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>inf</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



Similarly, using the Pandas API, we can select entries based on the "antecedents" or "consequents" columns:


```python
rules[rules['antecedents'] == {'Eggs', 'Kidney Beans'}]
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
      <th>antecedent_len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>(Kidney Beans, Eggs)</td>
      <td>(Onion)</td>
      <td>0.8</td>
      <td>0.6</td>
      <td>0.6</td>
      <td>0.75</td>
      <td>1.25</td>
      <td>0.12</td>
      <td>1.6</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**Frozensets**

Note that the entries in the "itemsets" column are of type `frozenset`, which is built-in Python type that is similar to a Python `set` but immutable, which makes it more efficient for certain query or comparison operations (https://docs.python.org/3.6/library/stdtypes.html#frozenset). Since `frozenset`s are sets, the item order does not matter. I.e., the query

`rules[rules['antecedents'] == {'Eggs', 'Kidney Beans'}]`
    
is equivalent to any of the following three

- `rules[rules['antecedents'] == {'Kidney Beans', 'Eggs'}]`
- `rules[rules['antecedents'] == frozenset(('Eggs', 'Kidney Beans'))]`
- `rules[rules['antecedents'] == frozenset(('Kidney Beans', 'Eggs'))]`




## Example 3 -- Frequent Itemsets with Incomplete Antecedent and Consequent Information

Most metrics computed by `association_rules` depends on the consequent and antecedent support score of a given rule provided in the frequent itemset input DataFrame. Consider the following example:


```python
import pandas as pd

dict = {'itemsets': [['177', '176'], ['177', '179'],
                     ['176', '178'], ['176', '179'],
                     ['93', '100'], ['177', '178'],
                     ['177', '176', '178']],
        'support':[0.253623, 0.253623, 0.217391,
                   0.217391, 0.181159, 0.108696, 0.108696]}

freq_itemsets = pd.DataFrame(dict)
freq_itemsets
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
      <th>itemsets</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[177, 176]</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[177, 179]</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[176, 178]</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[176, 179]</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[93, 100]</td>
      <td>0.181159</td>
    </tr>
    <tr>
      <th>5</th>
      <td>[177, 178]</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>6</th>
      <td>[177, 176, 178]</td>
      <td>0.108696</td>
    </tr>
  </tbody>
</table>
</div>



Note that this is a "cropped" DataFrame that doesn't contain the support values of the item subsets. This can create problems if we want to compute the association rule metrics for, e.g., `176 => 177`.

For example, the confidence is computed as

$$\text{confidence}(A\rightarrow C) = \frac{\text{support}(A\rightarrow C)}{\text{support}(A)}, \;\;\; \text{range: } [0, 1]$$

But we do not have $\text{support}(A)$. All we know about "A"'s support is that it is at least 0.253623.

In these scenarios, where not all metric's can be computed, due to incomplete input DataFrames, you can use the `support_only=True` option, which will only compute the support column of a given rule that does not require as much info:

$$\text{support}(A\rightarrow C) = \text{support}(A \cup C), \;\;\; \text{range: } [0, 1]$$


"NaN's" will be assigned to all other metric columns:


```python
from mlxtend.frequent_patterns import association_rules

res = association_rules(freq_itemsets, support_only=True, min_threshold=0.1)
res
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>antecedent support</th>
      <th>consequent support</th>
      <th>support</th>
      <th>confidence</th>
      <th>lift</th>
      <th>leverage</th>
      <th>conviction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(176)</td>
      <td>(177)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.253623</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(177)</td>
      <td>(176)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.253623</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(179)</td>
      <td>(177)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.253623</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(177)</td>
      <td>(179)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.253623</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(176)</td>
      <td>(178)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.217391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(178)</td>
      <td>(176)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.217391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(179)</td>
      <td>(176)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.217391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(176)</td>
      <td>(179)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.217391</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(93)</td>
      <td>(100)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.181159</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(100)</td>
      <td>(93)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.181159</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(177)</td>
      <td>(178)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(178)</td>
      <td>(177)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(176, 177)</td>
      <td>(178)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(176, 178)</td>
      <td>(177)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(177, 178)</td>
      <td>(176)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(176)</td>
      <td>(177, 178)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(177)</td>
      <td>(176, 178)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(178)</td>
      <td>(176, 177)</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.108696</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



To clean up the representation, you may want to do the following:


```python
res = res[['antecedents', 'consequents', 'support']]
res
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
      <th>antecedents</th>
      <th>consequents</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>(176)</td>
      <td>(177)</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(177)</td>
      <td>(176)</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>2</th>
      <td>(179)</td>
      <td>(177)</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>3</th>
      <td>(177)</td>
      <td>(179)</td>
      <td>0.253623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>(176)</td>
      <td>(178)</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>5</th>
      <td>(178)</td>
      <td>(176)</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>6</th>
      <td>(179)</td>
      <td>(176)</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>7</th>
      <td>(176)</td>
      <td>(179)</td>
      <td>0.217391</td>
    </tr>
    <tr>
      <th>8</th>
      <td>(93)</td>
      <td>(100)</td>
      <td>0.181159</td>
    </tr>
    <tr>
      <th>9</th>
      <td>(100)</td>
      <td>(93)</td>
      <td>0.181159</td>
    </tr>
    <tr>
      <th>10</th>
      <td>(177)</td>
      <td>(178)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>11</th>
      <td>(178)</td>
      <td>(177)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>12</th>
      <td>(176, 177)</td>
      <td>(178)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>13</th>
      <td>(176, 178)</td>
      <td>(177)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>14</th>
      <td>(177, 178)</td>
      <td>(176)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>15</th>
      <td>(176)</td>
      <td>(177, 178)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>16</th>
      <td>(177)</td>
      <td>(176, 178)</td>
      <td>0.108696</td>
    </tr>
    <tr>
      <th>17</th>
      <td>(178)</td>
      <td>(176, 177)</td>
      <td>0.108696</td>
    </tr>
  </tbody>
</table>
</div>



## API


*association_rules(df, metric='confidence', min_threshold=0.8, support_only=False)*

Generates a DataFrame of association rules including the
metrics 'score', 'confidence', and 'lift'

**Parameters**

- `df` : pandas DataFrame

    pandas DataFrame of frequent itemsets
    with columns ['support', 'itemsets']


- `metric` : string (default: 'confidence')

    Metric to evaluate if a rule is of interest.
**Automatically set to 'support' if `support_only=True`.**
    Otherwise, supported metrics are 'support', 'confidence', 'lift',

'leverage', and 'conviction'
    These metrics are computed as follows:

    - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]

    - confidence(A->C) = support(A+C) / support(A), range: [0, 1]

    - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]

    - leverage(A->C) = support(A->C) - support(A)*support(C),
    range: [-1, 1]

    - conviction = [1 - support(C)] / [1 - confidence(A->C)],
    range: [0, inf]



- `min_threshold` : float (default: 0.8)

    Minimal threshold for the evaluation metric,
    via the `metric` parameter,
    to decide whether a candidate rule is of interest.


- `support_only` : bool (default: False)

    Only computes the rule support and fills the other
    metric columns with NaNs. This is useful if:

    a) the input DataFrame is incomplete, e.g., does
    not contain support values for all rule antecedents
    and consequents

    b) you simply want to speed up the computation because
    you don't need the other metrics.

**Returns**

pandas DataFrame with columns "antecedents" and "consequents"
    that store itemsets, plus the scoring metric columns:
    "antecedent support", "consequent support",
    "support", "confidence", "lift",
    "leverage", "conviction"
    of all rules for which
    metric(rule) >= min_threshold.
    Each entry in the "antecedents" and "consequents" columns are
    of type `frozenset`, which is a Python built-in type that
    behaves similarly to sets except that it is immutable
    (For more info, see
    https://docs.python.org/3.6/library/stdtypes.html#frozenset).

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)


