# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.stats
from itertools import combinations


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

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/

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


def mcnemar_tables(y_target, *y_model_predictions):
    """
    Compute multiple 2x2 contigency tables for McNemar's
    test or Cochran's Q test.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.

    y_model_predictions : array-like, shape=[n_samples]
        Predicted class labels for a model.

    Returns
    ----------

    tables : dict
        Dictionary of NumPy arrays with shape=[2, 2]. Each dictionary
        key names the two models to be compared based on the order the
        models were passed as `*y_model_predictions`. The number of
        dictionary entries is equal to the number of pairwise combinations
        between the m models, i.e., "m choose 2."

        For example the following target array (containing the true labels)
        and 3 models

        - y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        - y_mod0 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        - y_mod0 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
        - y_mod0 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0])

        would result in the following dictionary:


        {'model_0 vs model_1': array([[ 4.,  1.],
                                      [ 2.,  3.]]),
         'model_0 vs model_2': array([[ 3.,  0.],
                                      [ 3.,  4.]]),
         'model_1 vs model_2': array([[ 3.,  0.],
                                      [ 2.,  5.]])}

        Each array is structured in the following way:

        - tb[0, 0]: # of samples that both models predicted correctly
        - tb[0, 1]: # of samples that model a got right and model b got wrong
        - tb[1, 0]: # of samples that model b got right and model a got wrong
        - tb[1, 1]: # of samples that both models predicted incorrectly

    Examples
    -----------

    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/

    """
    model_lens = set()
    y_model_predictions = list(y_model_predictions)
    for ary in ([y_target] + y_model_predictions):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')
        model_lens.add(ary.shape[0])

    if len(model_lens) > 1:
        raise ValueError('Each prediction array must have the '
                         'same number of samples.')

    num_models = len(y_model_predictions)

    if num_models < 2:
        raise ValueError('Provide at least 2 model prediction arrays.')

    tables = {}

    for comb in combinations(range(num_models), 2):

        tb = np.zeros((2, 2))
        model1_vs_true = (y_target == y_model_predictions[comb[0]]).astype(int)
        model2_vs_true = (y_target == y_model_predictions[comb[1]]).astype(int)
        plus_true = model1_vs_true + model2_vs_true
        minus_true = model1_vs_true - model2_vs_true

        tb[0, 0] = np.sum(plus_true == 2)
        tb[1, 1] = np.sum(plus_true == 0)
        tb[1, 0] = np.sum(minus_true == 1)
        tb[0, 1] = np.sum(minus_true == -1)

        name_str = 'model_%s vs model_%s' % (comb[0], comb[1])
        tables[name_str] = tb

    return tables


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

    Examples
    -----------

    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/

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
        chi2 = min(b, c)
        p = min(scipy.stats.binom.cdf(chi2, b + c, .5) * 2., 1.)

    return chi2, p
