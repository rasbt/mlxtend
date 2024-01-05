# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlxtend.math import factorial, num_combinations, num_permutations


def test_factorial():
    assert factorial(1) == 1
    assert factorial(3) == 6


def test_num_combinations():
    assert num_combinations(n=20, k=8, with_replacement=False) == 125970
    assert num_combinations(n=20, k=8, with_replacement=True) == 2220075
    print(num_combinations(n=300, k=10))
    assert num_combinations(n=300, k=10, with_replacement=False) == 1398320233241701770


def test_num_permutations():
    assert num_permutations(n=20, k=8, with_replacement=False) == 5079110400
    assert num_permutations(n=20, k=8, with_replacement=True) == 25600000000
