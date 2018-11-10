mlxtend version: 0.14.0dev 
## factorial

*factorial(n)*

None




## num_combinations

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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/math/num_combinations/](http://rasbt.github.io/mlxtend/user_guide/math/num_combinations/)




## num_permutations

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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/math/num_permutations/](http://rasbt.github.io/mlxtend/user_guide/math/num_permutations/)




## vectorspace_dimensionality

*vectorspace_dimensionality(ary)*

Computes the hyper-volume spanned by a vector set

**Parameters**

- `ary` : array-like, shape=[num_vectors, num_vectors]

    An orthogonal set of vectors (arranged as columns in a matrix)

**Returns**

- `dimensions` : int

    An integer indicating the "dimensionality" hyper-volume spanned by
    the vector set




## vectorspace_orthonormalization

*vectorspace_orthonormalization(ary, eps=1e-13)*

Transforms a set of column vectors to a orthonormal basis.

Given a set of orthogonal vectors, this functions converts such
column vectors, arranged in a matrix, into orthonormal basis
vectors.

**Parameters**

- `ary` : array-like, shape=[num_vectors, num_vectors]

    An orthogonal set of vectors (arranged as columns in a matrix)


- `eps` : float (default: 1e-13)

    A small tolerance value to determine whether
    the vector norm is zero or not.

**Returns**

- `arr` : array-like, shape=[num_vectors, num_vectors]

    An orthonormal set of vectors (arranged as columns)




