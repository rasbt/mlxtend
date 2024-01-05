# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np

from mlxtend.math import vectorspace_dimensionality, vectorspace_orthonormalization


def test_vectorspace_orthonormalization():
    a1 = np.array([[2, 0, 4, 12], [0, 2, 16, 4], [4, 16, 6, 2], [2, -12, 4, 6]])

    expect1 = np.array(
        [
            [0.40824829, -0.1814885, 0.04982278, 0.89325973],
            [0.0, 0.1088931, 0.99349591, -0.03328918],
            [0.81649658, 0.50816781, -0.06462163, -0.26631346],
            [0.40824829, -0.83484711, 0.07942048, -0.36063281],
        ]
    )

    np.testing.assert_array_almost_equal(
        vectorspace_orthonormalization(a1), expect1, decimal=7
    )

    np.testing.assert_array_almost_equal(
        vectorspace_orthonormalization(a1 / 2), expect1, decimal=7
    )


def test_vectorspace_dimensionality():
    a = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert vectorspace_dimensionality(a) == 1

    a[:, 1] *= 2
    a[:, 2] *= 4

    assert vectorspace_dimensionality(a) == 1

    a[:, 1] += np.array([3, 34, 99])

    assert vectorspace_dimensionality(a) == 2

    b = np.array([[1, 2, 3], [15, 1, 3]])
    assert vectorspace_dimensionality(b) == 2
