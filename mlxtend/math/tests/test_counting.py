# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlxtend.math import factorial
from mlxtend.math import num_permutations
from mlxtend.math import num_combinations


def test_factorial():
    assert(factorial(1) == 1)
    assert(factorial(3) == 6)


def test_num_combinations():
    assert(num_combinations(n=20, k=8, with_replacement=False) == 125970)
    assert(num_combinations(n=20, k=8, with_replacement=True) == 2220075)


def test_num_permutations():
    assert(num_permutations(n=20, k=8, with_replacement=False) == 5079110400)
    assert(num_permutations(n=20, k=8, with_replacement=True) == 25600000000)
