# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import chi2_contingency

from mlxtend.classifier import OneRClassifier


@pytest.mark.parametrize("reverse", [False, True])
def test_chi_squared_tie_selects_lowest_pvalue(reverse):
    # Both features make two errors. The second has the smaller chi-square p-value.
    features = np.array(
        [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 1], [1, 1]]
    )
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    if reverse:
        features = features[:, ::-1]
    expected_index = 0 if reverse else 1
    expected_pvalue = chi2_contingency([[4, 0], [2, 2]]).pvalue

    model = OneRClassifier(resolve_ties="chi-squared").fit(features, targets)

    assert model.feature_idx_ == expected_index
    assert_allclose(model.p_value_, expected_pvalue)
    assert model.prediction_dict_["total error"] == 2
    assert_array_equal(model.predict(features), features[:, expected_index])


def test_first_tie_resolution_remains_in_feature_order():
    features = np.array(
        [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [1, 0], [1, 1], [1, 1]]
    )
    targets = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model = OneRClassifier(resolve_ties="first").fit(features, targets)

    assert model.feature_idx_ == 0
    assert_array_equal(model.predict(features), features[:, 0])
