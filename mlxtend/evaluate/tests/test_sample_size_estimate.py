# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Prajwal Kafle
#
# License: BSD 3 clause

from mlxtend.evaluate import sample_size_estimate


def test_one_sided_binomial_proportion_estimates():
    n = sample_size_estimate.binomial_proportions(15, 20, 95, 80,
                                                  test='one-sided')
    assert 1892 == n


def test_two_sided_binomial_proportion_estimates():
    n = sample_size_estimate.binomial_proportions(15, 20, 95, 80)
    assert 2403 == n
