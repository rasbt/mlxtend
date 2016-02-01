# Compute the Number of Combinations

A function to calculate the number of combinations for creating subsequences of *k* elements out of a sequence with *n* elements.

> from mlxtend.math import num_combinations

# Overview

Combinations are selections of items from a collection regardless of the order in which they appear (in contrast to permutations). For example, let's consider a combination of 3 elements (k=3) from a collection of 5 elements (n=5): 

- collection: {1, 2, 3, 4, 5}
- combination 1a: {1, 3, 5} 
- combination 1b: {1, 5, 3}
- combination 1c: {3, 5, 1}
- ...
- combination 2: {1, 3, 4}

In the example above the combinations 1a, 1b, and 1c, are the "same combination" and counted as "1 possible way to combine items 1, 3, and 5" -- in combinations, the order does not matter.



The number of ways to combine elements (**without replacement**)  from a collection with size *n* into subsets of size *k* is computed via the binomial coefficient ("*n* choose *k*"):

\\[ \begin{pmatrix} 
n  \\
k 
\end{pmatrix} = \frac{n(n-1)\ldots(n-k+1)}{k(k-1)\dots1} = \frac{n!}{k!(n-k)!}  \\]

To compute the number of combinations **with replacement**, the following, alternative equation 
is used ("*n* multichoose *k*"):

\\[ \bigg(\begin{pmatrix} 
n  \\
k 
\end{pmatrix}\bigg) = \begin{pmatrix} 
n + k -1  \\
k 
\end{pmatrix}  \\]

### References

- [https://en.wikipedia.org/wiki/Combination](https://en.wikipedia.org/wiki/Combination)

# Examples

## Example 1 - Compute the number of combinations


```python
from mlxtend.math import num_combinations

c = num_combinations(n=20, k=8, with_replacement=False)
print('Number of ways to combine 20 elements into 8 subelements: %d' % c)
```

    Number of ways to combine 20 elements into 8 subelements: 125970



```python
from mlxtend.math import num_combinations

c = num_combinations(n=20, k=8, with_replacement=True)
print('Number of ways to combine 20 elements into 8 subelements (with replacement): %d' % c)
```

    Number of ways to combine 20 elements into 8 subelements (with replacement): 2220075


## Example 2 - A progress tracking use-case

It is often quite useful to track the progress of a computational expensive tasks to estimate its runtime. Here, the `num_combination` function can be used to compute the maximum number of loops of a `combinations` iterable from itertools:


```python
import itertools
import sys
import time
from mlxtend.math import num_combinations

items = {1, 2, 3, 4, 5, 6, 7, 8}
max_iter = num_combinations(n=len(items), k=3, 
                            with_replacement=False)

for idx, i in enumerate(itertools.combinations(items, r=3)):
    # do some computation with itemset i
    time.sleep(0.1)
    sys.stdout.write('\rProgress: %d/%d' % (idx + 1, max_iter))
    sys.stdout.flush()
```

    Progress: 56/56

# API


*num_combinations(n, k, with_replacement=False)*

Function to calculate the number of possible combinations.

**Parameters**

- `n` : `int`

    Total number of items.

- `k` : `int`

    Number of elements of the target itemset.

- `with_replacement` : `bool` (default: False)

    Allows repeated elements if True.

**Returns**

- `comb` : `int`

    Number of possible combinations.


