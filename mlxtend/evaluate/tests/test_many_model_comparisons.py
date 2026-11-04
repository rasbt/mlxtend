# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import chi2, f

from mlxtend.evaluate import cochrans_q, ftest


@pytest.mark.parametrize("copies", [1, 20])
def test_model_comparisons_with_many_prediction_arrays(copies):
    # Four classifiers have 5, 3, 3 and 1 correct answers, respectively.
    # Repeating them leaves the F statistic unchanged and scales Q by
    # (number of classifiers - 1) / 3.
    target = np.zeros(6, dtype=int)
    predictions = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 0, 1],
        ]
    ).T
    models = list(predictions) * copies
    degrees = len(models) - 1

    q_statistic, q_pvalue = cochrans_q(target, *models)
    f_statistic, f_pvalue = ftest(target, *models)

    assert_allclose(q_statistic, 16 * degrees / 7)
    assert_allclose(q_pvalue, chi2.sf(16 * degrees / 7, degrees))
    assert_allclose(f_statistic, 40 / 13)
    assert_allclose(f_pvalue, f.sf(40 / 13, degrees, degrees * len(target)))


@pytest.mark.parametrize("comparison", [cochrans_q, ftest])
def test_model_comparison_is_invariant_to_sample_and_model_order(comparison):
    rng = np.random.RandomState(7)
    target = rng.randint(0, 3, 17)
    predictions = rng.randint(0, 3, (40, 17))
    order = rng.permutation(17)

    expected = comparison(target, *predictions)
    actual = comparison(target[order], *predictions[::-1, order])

    assert_allclose(actual, expected)
