# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.stats


def mcnemar_table(y_target, y_model1, y_model2):
    """
    Compute a 2x2 contigency table for McNemar's test.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.
    y_model1 : array-like, shape=[n_samples]
        Predicted class labels from model as 1D NumPy array.
    y_model2 : array-like, shape=[n_samples]
        Predicted class labels from model 2 as 1D NumPy array.

    Returns
    ----------
    tb : array-like, shape=[2, 2]
       2x2 contingency table with the following contents:
       a: tb[0, 0]: # of samples that both models predicted correctly
       b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
       c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
       d: tb[1, 1]: # of samples that both models predicted incorrectly
    """
    for ary in (y_target, y_model1, y_model2):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')

    if y_target.shape[0] != y_model1.shape[0]:
        raise ValueError('y_target and y_model1 contain a different number'
                         ' of elements.')

    if y_target.shape[0] != y_model2.shape[0]:
        raise ValueError('y_target and y_model2 contain a different number'
                         ' of elements.')

    m1_vs_true = (y_target == y_model1).astype(int)
    m2_vs_true = (y_target == y_model2).astype(int)

    plus_true = m1_vs_true + m2_vs_true
    minus_true = m1_vs_true - m2_vs_true

    tb = np.zeros((2, 2), dtype=int)

    tb[0, 0] = np.sum(plus_true == 2)
    tb[1, 1] = np.sum(plus_true == 0)
    tb[1, 0] = np.sum(minus_true == 1)
    tb[0, 1] = np.sum(minus_true == -1)

    return tb


def mcnemar(ary, corrected=True, exact=False):
    """
    McNemar test for paired nominal data

    Parameters
    -----------
    ary : array-like, shape=[2, 2]
        2 x 2 contigency table (as returned by evaluate.mcnemar_table),
        where
        a: ary[0, 0]: # of samples that both models predicted correctly
        b: ary[0, 1]: # of samples that model 1 got right and model 2 got wrong
        c: ary[1, 0]: # of samples that model 2 got right and model 1 got wrong
        d: aryCell [1, 1]: # of samples that both models predicted incorrectly
    corrected : array-like, shape=[n_samples] (default: True)
        Uses Edward's continuity correction for chi-squared if `True`
    exact : bool, (default: False)
        If `True`, uses an exact binomial test comparing b to
        a binomial distribution with n = b + c and p = 0.5.
        It is highly recommended to use `exact=True` for sample sizes < 25
        since chi-squared is not well-approximated
        by the chi-squared distribution!

    Returns
    -----------
    chi2, p : float or None, float
        Returns the chi-squared value and the p-value;
        if `exact=True` (default: `False`), `chi2` is `None`

    """

    if not ary.shape == (2, 2):
        raise ValueError('Input array must be a 2x2 array.')

    b = ary[0, 1]
    c = ary[1, 0]
    n = b + c

    if not exact:
        if corrected:
            chi2 = (abs(ary[0, 1] - ary[1, 0]) - 1.0)**2 / float(n)
        else:
            chi2 = (ary[0, 1] - ary[1, 0])**2 / float(n)
        p = scipy.stats.distributions.chi2.sf(chi2, 1)

    else:
        p = 2. * sum([scipy.stats.binom.pmf(k=i, n=n, p=0.5, loc=0)
                      for i in range(b, n)])
        chi2 = None

    return chi2, p
