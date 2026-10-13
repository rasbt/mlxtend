# mlxtend Machine Learning Library Extensions
# Tests for calibration metrics (ECE, MCE, reliability table).
#
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from mlxtend.evaluate import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
)


def test_perfect_calibration_is_zero():
    # Confidence 1.0 and always correct -> no gap.
    y = np.array([0, 1, 1, 0])
    proba = np.array([0.0, 1.0, 1.0, 0.0])
    assert expected_calibration_error(y, proba) == pytest.approx(0.0)
    assert maximum_calibration_error(y, proba) == pytest.approx(0.0)


def test_known_value_binary():
    # Confidence 0.9 everywhere, accuracy 0.5 -> ECE = 0.4.
    y = np.array([1, 1, 0, 0])
    proba = np.array([0.9, 0.9, 0.9, 0.9])
    assert expected_calibration_error(y, proba) == pytest.approx(0.4)


def test_overconfident_wrong_predictions():
    # Always 100% confident and always wrong -> worst possible ECE.
    y = np.array([0, 0, 1, 1])
    proba = np.array([1.0, 1.0, 0.0, 0.0])
    assert expected_calibration_error(y, proba) == pytest.approx(1.0)


def test_hand_computed_two_bins():
    # Bin [0, .5]: none (confidence is always >= .5).
    # Bin (.5, 1]: conf {0.6, 0.8} -> avg 0.7; correct {1, 0} -> acc 0.5.
    y = np.array([1, 0])
    proba = np.array([0.6, 0.8])
    t = reliability_table(y, proba, n_bins=2)
    assert t["count"].tolist() == [2]
    assert t["avg_confidence"][0] == pytest.approx(0.7)
    assert t["accuracy"][0] == pytest.approx(0.5)
    assert expected_calibration_error(y, proba, n_bins=2) == pytest.approx(0.2)


def test_multiclass_2d_proba():
    y = np.array([0, 1, 2])
    proba = np.array([[0.7, 0.2, 0.1],
                      [0.1, 0.7, 0.2],
                      [0.7, 0.2, 0.1]])  # last one is wrong
    # All confidences 0.7, accuracy 2/3 -> ECE = |2/3 - 0.7|.
    assert expected_calibration_error(y, proba) == pytest.approx(0.7 - 2 / 3)


def test_binary_1d_and_2d_agree():
    rng = np.random.RandomState(0)
    p1 = rng.rand(200)
    y = rng.randint(0, 2, 200)
    p2 = np.column_stack([1 - p1, p1])
    assert expected_calibration_error(y, p1) == pytest.approx(
        expected_calibration_error(y, p2))


def test_mce_at_least_ece():
    rng = np.random.RandomState(1)
    p = rng.rand(500)
    y = rng.randint(0, 2, 500)
    assert maximum_calibration_error(y, p) >= expected_calibration_error(y, p)


def test_table_counts_sum_to_n():
    rng = np.random.RandomState(2)
    p = rng.rand(321)
    y = rng.randint(0, 2, 321)
    for strategy in ("uniform", "quantile"):
        t = reliability_table(y, p, n_bins=7, strategy=strategy)
        assert t["count"].sum() == 321


def test_quantile_bins_are_balanced():
    rng = np.random.RandomState(3)
    p = rng.rand(1000)
    y = rng.randint(0, 2, 1000)
    t = reliability_table(y, p, n_bins=5, strategy="quantile")
    assert t["count"].max() - t["count"].min() <= 2


def test_real_models():
    # Logistic regression is usually better calibrated than Naive Bayes.
    X, y = make_classification(n_samples=3000, n_features=20,
                               n_informative=5, random_state=0)
    Xtr, Xte, ytr, yte = X[:2000], X[2000:], y[:2000], y[2000:]
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    nb = GaussianNB().fit(Xtr, ytr)
    ece_lr = expected_calibration_error(yte, lr.predict_proba(Xte))
    ece_nb = expected_calibration_error(yte, nb.predict_proba(Xte))
    assert 0.0 <= ece_lr <= 1.0
    assert ece_lr < ece_nb


@pytest.mark.parametrize("y, proba, msg", [
    ([0, 1], [0.5], "same number of samples"),
    ([0, 1], [0.5, 1.5], "range"),
    ([0, 2], [0.5, 0.5], "only 0 and 1"),
    ([0, 1], [[0.5, 0.6], [0.5, 0.5]], "sum to 1"),
    ([0, 3], [[0.5, 0.5], [0.5, 0.5]], "n_classes"),
    ([], [], "at least one"),
    ([0, 1], [np.nan, 0.5], "NaN"),
])
def test_invalid_inputs(y, proba, msg):
    with pytest.raises(ValueError, match=msg):
        expected_calibration_error(y, proba)


def test_invalid_params():
    with pytest.raises(ValueError, match="n_bins"):
        expected_calibration_error([0, 1], [0.2, 0.8], n_bins=0)
    with pytest.raises(ValueError, match="strategy"):
        expected_calibration_error([0, 1], [0.2, 0.8], strategy="foo")
