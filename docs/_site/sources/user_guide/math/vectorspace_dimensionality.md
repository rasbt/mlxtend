# Vectorspace Dimensionality

A function to compute the number of dimensions a set of vectors (arranged as columns in a matrix) spans.

> from mlxtend.math import vectorspace_dimensionality

## Overview

Given a set of vectors, arranged as columns in a matrix, the `vectorspace_dimensionality` computes the number of dimensions (i.e., hyper-volume) that the vectorspace spans using the Gram-Schmidt process [1]. In particular, since the Gram-Schmidt process yields vectors that are zero or normalized to 1 (i.e., an orthonormal vectorset if the input was a set of linearly independent vectors), the sum of the vector norms corresponds to the number of dimensions of a vectorset. 

### References

- [1] https://en.wikipedia.org/wiki/Gram–Schmidt_process

## Example 1 - Compute the dimensions of a vectorspace

Let's assume we have the two basis vectors $x=[1 \;\;\; 0]^T$ and $y=[0\;\;\; 1]^T$ as columns in a matrix. Due to the linear independence of the two vectors, the space that they span is naturally a plane (2D space):


```python
import numpy as np
from mlxtend.math import vectorspace_dimensionality


a = np.array([[1, 0],
              [0, 1]])

vectorspace_dimensionality(a)
```




    2



However, if one vector is a linear combination of the other, it's intuitive to see that the space the vectorset describes is merely a line, aka a 1D space:


```python
b = np.array([[1, 2],
              [0, 0]])

vectorspace_dimensionality(a)
```




    2



If 3 vectors are all linearly independent of each other, the dimensionality of the vector space is a volume (i.e., a 3D space):


```python
d = np.array([[1, 9,  1],
              [3, 2,  2],
              [5, 4,  3]])

vectorspace_dimensionality(d)
```




    3



Again, if a pair of vectors is linearly dependent (here: the 1st and the 2nd row), this reduces the dimensionality by 1:


```python
c = np.array([[1, 2,  1],
              [3, 6,  2],
              [5, 10, 3]])

vectorspace_dimensionality(c)
```




    2



## API


*vectorspace_dimensionality(ary)*

Computes the hyper-volume spanned by a vector set

**Parameters**

- `ary` : array-like, shape=[num_vectors, num_vectors]

    A set of vectors (arranged as columns in a matrix)

**Returns**

- `dimensions` : int

    An integer indicating the "dimensionality" hyper-volume spanned by
    the vector set


