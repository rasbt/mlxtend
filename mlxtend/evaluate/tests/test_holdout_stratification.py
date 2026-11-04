# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_array_equal
from sklearn.model_selection import train_test_split

from mlxtend.evaluate import RandomHoldoutSplit


@pytest.mark.parametrize(
    "targets",
    [np.arange(12, dtype=float) / 7, np.array([0] * 11 + [1])],
)
def test_unstratified_holdout_accepts_continuous_and_rare_targets(targets):
    features = np.zeros((targets.size, 2))
    expected = train_test_split(
        np.arange(targets.size), test_size=0.25, random_state=17
    )

    actual = list(
        RandomHoldoutSplit(valid_size=0.25, random_seed=17).split(features, targets)
    )

    assert len(actual) == 1
    for result, reference in zip(actual[0], expected):
        assert_array_equal(result, reference)


@pytest.mark.parametrize("stratify", [False, True])
def test_holdout_matches_requested_stratification(stratify):
    targets = np.repeat([0, 1, 2], 10)
    features = np.zeros((targets.size, 2))
    expected = train_test_split(
        np.arange(targets.size),
        test_size=0.3,
        random_state=23,
        stratify=targets if stratify else None,
    )

    actual = next(
        RandomHoldoutSplit(valid_size=0.3, random_seed=23, stratify=stratify).split(
            features, targets
        )
    )

    for result, reference in zip(actual, expected):
        assert_array_equal(result, reference)
