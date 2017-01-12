# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Functions for different counting operations.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)


def num_combinations(n, k, with_replacement=False):
    """
    Function to calculate the number of possible combinations.

    Parameters
    ----------
    n : `int`
        Total number of items.
    k : `int`
        Number of elements of the target itemset.
    with_replacement : `bool` (default: False)
        Allows repeated elements if True.

    Returns
    ----------
    comb : `int`
        Number of possible combinations.
    """
    if with_replacement:
        numerator = factorial(n + k - 1)
        denominator = factorial(k) * factorial(n-1)
    else:
        numerator = factorial(n)
        denominator = factorial(k) * factorial(n-k)
    comb = int(numerator/denominator)
    return comb


def num_permutations(n, k, with_replacement=False):
    """
    Function to calculate the number of possible permutations.

    Parameters
    ----------
    n : `int`
      Total number of items.
    k : `int`
      Number of elements of the target itemset.
    with_replacement : `bool`
      Allows repeated elements if True.

    Returns
    ----------
    permut : `int`
      Number of possible permutations.

    """
    if with_replacement:
        permut = n**k
    else:
        numerator = factorial(n)
        denominator = factorial(n-k)
        permut = int(numerator/denominator)
    return permut
