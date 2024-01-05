# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Functions for different linear algebra operations.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def vectorspace_orthonormalization(ary, eps=1e-13):  # method='gram-schmidt',
    """Transforms a set of column vectors to a orthonormal basis.

    Given a set of orthogonal vectors, this functions converts such
    column vectors, arranged in a matrix, into orthonormal basis
    vectors.

    Parameters
    ----------
    ary : array-like, shape=[num_vectors, num_vectors]
        An orthogonal set of vectors (arranged as columns in a matrix)

    eps : float (default: 1e-13)
        A small tolerance value to determine whether
        the vector norm is zero or not.

    Returns
    ----------
    arr : array-like, shape=[num_vectors, num_vectors]
        An orthonormal set of vectors (arranged as columns)

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/math/vectorspace_orthonormalization/

    """

    # Gram-Schmidt Process
    #  1) col i for i = 0: keep & normalize:
    #  2) col i for i > 1:
    #   2a) project column i+1 onto i
    #   2b) subtract column i from projection vector
    #   2c) Normalize if linearly independent,
    #       and set to zero otherwise

    arr = ary.astype(np.float_).copy()

    for i in range(arr.shape[1]):
        for j in range(i):
            # 2a) & 2b)
            arr[:, i] -= np.dot(arr[:, i], arr[:, j]) * arr[:, j]
        # 2c
        tmp = np.linalg.norm(arr[:, i])
        is_linearly_indepedent = tmp > eps
        if is_linearly_indepedent:
            arr[:, i] /= tmp
        else:
            arr[:, i] = np.zeros(arr[:, i].shape)

    # elif method == 'qr-factorization':
    #    Q, R = np.linalg.qr(ary)
    #    arr = Q

    # QR factorization is not used here because of non-useful
    # results in cases of linear dependence

    # else:
    #     raise ValueError("Method must be 'gram-schmidt'"
    #                      "or 'qr-factorization'")

    return arr


def vectorspace_dimensionality(ary):
    """Computes the hyper-volume spanned by a vector set

    Parameters
    ----------
    ary : array-like, shape=[num_vectors, num_vectors]
        An orthogonal set of vectors (arranged as columns in a matrix)

    Returns
    ----------
    dimensions : int
        An integer indicating the "dimensionality" hyper-volume spanned by
        the vector set

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/math/vectorspace_dimensionality/

    """
    # Note that since the vectors of
    # an orthonormal vectoset have unit length or are zero,
    # the sum of the individual
    # norms equals the dimensionality of that vector space
    return int(np.sum(np.linalg.norm(vectorspace_orthonormalization(ary), axis=0)))
