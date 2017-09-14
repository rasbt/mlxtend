# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from itertools import combinations
from nose.tools import nottest


# decorator to prevent nose to consider
# this as a unit test due to "test" in the name
@nottest
def permutation_test(x, y, alt_hypothesis='x != y', method='exact',
                     num_permutations=1000, seed=None):
    """
    Nonparametric permutation test

    Parameters
    -------------
    x : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the first sample
        (e.g., the treatment group).
    y : list or numpy array with shape (n_datapoints,)
        A list or 1D numpy array of the second sample
        (e.g., the control group).
    alt_hypothesis : str (default='x != y')
        The alternativ hypothesis. It 'x > y' or 'x < y',
        a one-sided permutation test is performerd to test, and
        a two-sided permutation test is performed if 'x != y' (default),
        to test the null hypothesis that both x and y are samples
        from the same population.
    method : 'approximate' or 'exact' (default: 'exact')
        If 'exact' (default), all possible permutations are considered.
        If 'approximate' the number of drawn samples is
        given by `num_permutations`.
        Note that 'exact' is typically not feasible unless the dataset
        size is relatively small.
    num_permutations : int (default: 1000)
        The number of permutation samples if `method='approximate'`.
    seed : int or None (default: None)
        The random seed for generating permutation samples if
        `method='approximate'`.

    Returns
    ----------
    p-value under the null hypothesis

    """
    if alt_hypothesis not in ('x > y', 'x < y', 'x != y'):
        raise AttributeError("alt_hypothesis be 'x > y',"
                             " 'y > x', or 'x != y', got %s" % alt_hypothesis)

    if method not in ('approximate', 'exact'):
        raise AttributeError('method must be "approximate"'
                             ' or "exact", got %s' % method)

    if isinstance(x, np.ndarray) and len(x.shape) != 1:
        raise AttributeError('x must be one-dimensional')

    if isinstance(y, np.ndarray) and len(y.shape) != 1:
        raise AttributeError('y must be one-dimensional')

    def one_sided_statistic(comb, x):
        # equivalent to np.mean(x) - np.mean(y)
        # where comb = np.vstack(x, y)
        sum_comb, sum_x = float(sum(comb)), float(sum(x))  # float for Py27
        return (sum_x / len(x) - (sum_comb - sum_x) / (len(comb) - len(x)))

    def two_sided_statistic(comb, x):
        # equivalent to np.abs(np.mean(x) - np.mean(y))
        # where comb = np.vstack(x, y)
        return np.abs(one_sided_statistic(comb, x))

    if alt_hypothesis == 'x > y':
        fun = one_sided_statistic
    elif alt_hypothesis == 'x < y':
        fun = one_sided_statistic
        x, y = y, x
    else:
        fun = two_sided_statistic

    rng = np.random.RandomState(seed)
    combined = np.hstack((x, y))
    reference_stat = fun(combined, x)

    more_extreme = 0.

    if method == 'exact':
        for count, perm in enumerate(combinations(combined, len(x)), 1):
            if fun(combined, perm) > reference_stat:
                more_extreme += 1.

    else:
        for count in range(num_permutations):
            rng.shuffle(combined)
            if fun(combined, combined[0:len(x)]) > reference_stat:
                more_extreme += 1.

    return more_extreme / count
