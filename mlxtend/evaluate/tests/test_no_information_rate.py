# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

from decimal import Decimal
from itertools import product

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

import mlxtend.evaluate.bootstrap_point632 as bootstrap_module
from mlxtend.evaluate import bootstrap_point632_score
from mlxtend.evaluate.bootstrap_point632 import accuracy, no_information_rate


@pytest.mark.parametrize(
    "targets, predictions",
    [
        ([0, 0, 1], [0, 1, 1]),
        ([-2, -2, 5, 10], [5, 10, 10]),
        (["cat", "cat", "dog"], ["dog", "cat"]),
        ([True, False, True], [False, False]),
        ([0, 1], [2, 3]),
        ([7], [7]),
        ([np.nan, 1.0, 1.0], [np.nan, 1.0]),
        ([np.nan], [np.nan]),
        ([1, 2], ["1", "3"]),
        ([2**53, 2**53 + 1], [float(2**53)]),
        ([None, "cat"], ["cat", None]),
        ([[0, 1], [1, 0]], [[1, 0], [1, 1]]),
        (np.arange(12)[::2], np.arange(8)[::2]),
    ],
)
def test_accuracy_no_information_matches_cartesian_product(targets, predictions):
    combinations = np.array(list(product(targets, predictions)))
    expected = accuracy(combinations[:, 0], combinations[:, 1])
    assert no_information_rate(targets, predictions, accuracy) == pytest.approx(
        expected
    )


def test_accuracy_no_information_avoids_cartesian_product(monkeypatch):
    def reject_product(*args):
        raise AssertionError("The default accuracy must not construct sample pairs")

    monkeypatch.setattr(bootstrap_module, "product", reject_product)
    targets = np.tile([0, 0, 1, 2], 25_000)
    predictions = np.tile([0, 1, 1, 2], 25_000)
    assert no_information_rate(targets, predictions, accuracy) == pytest.approx(0.3125)


def test_default_632plus_avoids_cartesian_product(monkeypatch):
    def reject_product(*args):
        raise AssertionError("The default accuracy must not construct sample pairs")

    monkeypatch.setattr(bootstrap_module, "product", reject_product)
    rng = np.random.RandomState(123)
    X = rng.normal(size=(1_000, 2))
    y = (X[:, 0] > 0).astype(int)
    scores = bootstrap_point632_score(
        DecisionTreeClassifier(max_depth=1, random_state=123),
        X,
        y,
        n_splits=2,
        method=".632+",
        random_seed=123,
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))


def test_custom_no_information_scorer_receives_all_pairs():
    targets = [2, 0, 2]
    predictions = [1, 2]
    expected = np.array(list(product(targets, predictions)))

    def custom_score(received_targets, received_predictions):
        np.testing.assert_array_equal(received_targets, expected[:, 0])
        np.testing.assert_array_equal(received_predictions, expected[:, 1])
        return 0.125

    assert no_information_rate(targets, predictions, custom_score) == 0.125


def test_object_no_information_preserves_decimal_nan_labels():
    # Decimal NaNs support equality but raise InvalidOperation during sorting.
    targets = [Decimal("NaN"), Decimal(1)]
    predictions = [Decimal(1), Decimal("NaN")]
    assert no_information_rate(targets, predictions, accuracy) == 0.25
