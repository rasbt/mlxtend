# Vectorspace Orthonormalization

A function that converts a set of linearly independent vectors to a set of orthonormal basis vectors.

> from mlxtend.math import vectorspace_orthonormalization

## Overview

The `vectorspace_orthonormalization` converts a set linearly independent vectors to a set of orthonormal basis vectors using the Gram-Schmidt process [1]. 

### References

- [1] https://en.wikipedia.org/wiki/Gram–Schmidt_process

## Example 1 - Convert a set of vector to an orthonormal basis

Note that to convert a set of linearly independent vectors into a set of orthonormal basis vectors, the `vectorspace_orthonormalization` function expects the vectors to be arranged as columns of a matrix (here: NumPy array). Please keep in mind that the `vectorspace_orthonormalization` function also works for non-linearly independent vector sets; however, the resulting vectorset won't be orthonormal as a result. An easy way to check whether all vectors in the input set are linearly independent is to use the `numpy.linalg.det` (determinant) function.


```python
import numpy as np
from mlxtend.math import vectorspace_orthonormalization
    
a = np.array([[2,   0,   4,  12],
              [0,   2,  16,   4],
              [4,  16,   6,   2],
              [2, -12,   4,   6]])


s = ''
if np.linalg.det(a) == 0.0:
    s = ' not'
print('Input vectors are%s linearly independent' % s)


vectorspace_orthonormalization(a)
```

    Input vectors are linearly independent





    array([[ 0.40824829, -0.1814885 ,  0.04982278,  0.89325973],
           [ 0.        ,  0.1088931 ,  0.99349591, -0.03328918],
           [ 0.81649658,  0.50816781, -0.06462163, -0.26631346],
           [ 0.40824829, -0.83484711,  0.07942048, -0.36063281]])



Note that scaling the inputs equally by a factor should leave the results unchanged:


```python
vectorspace_orthonormalization(a/2)
```




    array([[ 0.40824829, -0.1814885 ,  0.04982278,  0.89325973],
           [ 0.        ,  0.1088931 ,  0.99349591, -0.03328918],
           [ 0.81649658,  0.50816781, -0.06462163, -0.26631346],
           [ 0.40824829, -0.83484711,  0.07942048, -0.36063281]])



However, in case of linear dependence (the second column is a linear combination of the first column in the example below), the vector elements of one of the dependent vectors will become zero. (For a pair of linear dependent vectors, the one with the larger column index will be the one that's zero-ed.)


```python
a[:, 1] = a[:, 0] * 2
vectorspace_orthonormalization(a)
```




    array([[ 0.40824829,  0.        ,  0.04155858,  0.82364839],
           [ 0.        ,  0.        ,  0.99740596, -0.06501108],
           [ 0.81649658,  0.        , -0.04155858, -0.52008861],
           [ 0.40824829,  0.        ,  0.04155858,  0.21652883]])



## API


*vectorspace_orthonormalization(ary, eps=1e-13)*

Transforms a set of column vectors to a orthonormal basis.

Given a set of linearly independent vectors, this functions converts such
column vectors, arranged in a matrix, into orthonormal basis
vectors.

**Parameters**

- `ary` : array-like, shape=[num_vectors, num_vectors]

    A set of vectors (arranged as columns in a matrix)


- `eps` : float (default: 1e-13)

    A small tolerance value to determine whether
    the vector norm is zero or not.

**Returns**

- `arr` : array-like, shape=[num_vectors, num_vectors]

    An orthonormal set of vectors (arranged as columns)


