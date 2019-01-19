# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .counting import num_combinations
from .counting import num_permutations
from .counting import factorial
from .linalg import vectorspace_orthonormalization
from .linalg import vectorspace_dimensionality

__all__ = ["num_combinations", "num_permutations",
           "factorial", "vectorspace_orthonormalization",
           "vectorspace_dimensionality"]
