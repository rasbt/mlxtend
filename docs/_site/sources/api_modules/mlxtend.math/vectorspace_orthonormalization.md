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

