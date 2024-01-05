# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .counting import factorial, num_combinations, num_permutations
from .linalg import vectorspace_dimensionality, vectorspace_orthonormalization

__all__ = [
    "num_combinations",
    "num_permutations",
    "factorial",
    "vectorspace_orthonormalization",
    "vectorspace_dimensionality",
]
