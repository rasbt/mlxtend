# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.stats


def proportion_difference(proportion_1, proportion_2, n_1, n_2=None):
    """
    Computes the test statistic and p-value for a difference of
    proportions test.

    Parameters
    -----------
    proportion_1 : float
        The first proportion
    proportion_2 : float
        The second proportion
    n_1 : int
        The sample size of the first test sample
    n_2 : int or None (default=None)
        The sample size of the second test sample.
        If `None`, `n_1`=`n_2`.

    Returns
    -----------

    z, p : float or None, float
        Returns the z-score and the p-value


    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/

    """
    if n_2 is None:
        n_2 = n_1

    var_1 = proportion_1*(1. - proportion_1) / n_1
    var_2 = proportion_2*(1. - proportion_2) / n_2

    z = (proportion_1 - proportion_2) / np.sqrt(var_1 + var_2)
    p = scipy.stats.norm.cdf(z)

    return z, p
