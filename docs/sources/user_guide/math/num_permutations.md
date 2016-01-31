# Compute the Number of Permutations

A functions to calculate the number of permutations for creating subsequences of *k* elements out of a sequence with *n* elements.

> from mlxtend.math import num_permutations

# Overview

Permutations are selections of items from a collection with regard to the order in which they appear (in contrast to combinations). For example, let's consider a permutation of 3 elements (k=3) from a collection of 5 elements (n=5): 

- collection: {1, 2, 3, 4, 5}
- combination 1a: {1, 3, 5} 
- combination 1b: {1, 5, 3}
- combination 1c: {3, 5, 1}
- ...
- combination 2: {1, 3, 4}

In the example above the permutations 1a, 1b, and 1c, are the "same combination" but distinct permutations -- in combinations, the order does not matter, but in permutation it does matter.



The number of ways to combine elements (**without replacement**) from a collection with size *n* into subsets of size *k* is computed via the binomial coefficient ("*n* choose *k*"):

\\[ k!\begin{pmatrix} 
n  \\
k 
\end{pmatrix} = k! \cdot \frac{n!}{k!(n-k)!} = \frac{n!}{(n-k)!} \\]

To compute the number of permutations **with replacement**, we simply need to compute $n^k$.

### References

- [https://en.wikipedia.org/wiki/Permutation](https://en.wikipedia.org/wiki/Permutation)

### Related Topics

- [Compute the Number of Combinations](./num_combinations.html)

# Examples

## Example 1 - Compute the number of permutations


```python
from mlxtend.math import num_permutations

c = num_permutations(n=20, k=8, with_replacement=False)
print('Number of ways to permute 20 elements into 8 subelements: %d' % c)
```

    Number of ways to permute 20 elements into 8 subelements: 5079110400



```python
from mlxtend.math import num_permutations

c = num_permutations(n=20, k=8, with_replacement=True)
print('Number of ways to combine 20 elements into 8 subelements (with replacement): %d' % c)
```

    Number of ways to combine 20 elements into 8 subelements (with replacement): 25600000000


## Example 2 - A progress tracking use-case

It is often quite useful to track the progress of a computational expensive tasks to estimate its runtime. Here, the `num_combination` function can be used to compute the maximum number of loops of a `permutations` iterable from itertools:


```python
import itertools
import sys
import time
from mlxtend.math import num_permutations

items = {1, 2, 3, 4, 5, 6, 7, 8}
max_iter = num_permutations(n=len(items), k=3, 
                            with_replacement=False)

for idx, i in enumerate(itertools.permutations(items, r=3)):
    # do some computation with itemset i
    time.sleep(0.01)
    sys.stdout.write('\rProgress: %d/%d' % (idx + 1, max_iter))
    sys.stdout.flush()
```

    Progress: 336/336

# API


*num_permutations(n, k, with_replacement=False)*

Function to calculate the number of possible permutations.

**Parameters**

- `n` : `int`

    Total number of items.

- `k` : `int`

    Number of elements of the target itemset.

- `with_replacement` : `bool`

    Allows repeated elements if True.

**Returns**

- `permut` : `int`

    Number of possible permutations.


