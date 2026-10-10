# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from scipy.special import expit
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from mlxtend.evaluate import TemperatureScaling, expected_calibration_error


def _simulate(scale, n=20000, seed=0):
    """Labels drawn from true probabilities expit(z); predictions expit(scale*z).

    scale > 1 gives over-confident, scale < 1 under-confident predictions,
    so the ideal temperature is `scale`.
    """
    rng = np.random.RandomState(seed)
    z = rng.randn(n) * 2
    y = (rng.rand(n) < expit(z)).astype(int)
    return y, expit(scale * z)


def _log_loss(y, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


@pytest.mark.parametrize("scale", [3.0, 1.0, 0.4])
def test_recovers_known_temperature(scale):
    y, p = _simulate(scale)
    ts = TemperatureScaling().fit(p, y)
    assert ts.temperature_ == pytest.approx(scale, rel=0.1)


def test_overconfident_model_gets_better_calibrated():
    y, p = _simulate(3.0)
    p_cal = TemperatureScaling().fit_transform(p, y)
    assert expected_calibration_error(y, p_cal) < 0.2 * expected_calibration_error(y, p)


def test_underconfident_model_gets_better_calibrated():
    y, p = _simulate(0.4)
    p_cal = TemperatureScaling().fit_transform(p, y)
    assert expected_calibration_error(y, p_cal) < 0.2 * expected_calibration_error(y, p)


def test_log_loss_does_not_increase():
    y, p = _simulate(3.0)
    p_cal = TemperatureScaling().fit_transform(p, y)
    assert _log_loss(y, p_cal) <= _log_loss(y, p)


def test_class_labels_and_ranking_unchanged():
    y, p = _simulate(3.0)
    p_cal = TemperatureScaling().fit_transform(p, y)
    assert np.array_equal(p >= 0.5, p_cal >= 0.5)
    assert np.array_equal(
        np.argsort(p, kind="stable"), np.argsort(p_cal, kind="stable")
    )


def test_half_stays_half():
    y, p = _simulate(3.0)
    ts = TemperatureScaling().fit(p, y)
    assert ts.transform(np.array([0.5]))[0] == pytest.approx(0.5)


def test_extreme_probabilities_stay_finite():
    ts = TemperatureScaling().fit([0.0, 1.0, 0.2, 0.8], [0, 1, 1, 0])
    out = ts.transform(np.array([0.0, 1.0, 0.3]))
    assert np.all(np.isfinite(out))
    assert np.all((out >= 0) & (out <= 1))


def test_fit_returns_self_and_fit_transform_matches():
    y, p = _simulate(2.0, n=500)
    ts = TemperatureScaling()
    assert ts.fit(p, y) is ts
    np.testing.assert_allclose(
        ts.transform(p), TemperatureScaling().fit_transform(p, y)
    )


def test_output_shape_and_range():
    y, p = _simulate(2.0, n=300)
    out = TemperatureScaling().fit_transform(p, y)
    assert out.shape == p.shape
    assert np.all((out > 0) & (out < 1))


def test_perfectly_confident_correct_data_hits_lower_bound():
    ts = TemperatureScaling(bounds=(0.5, 10.0)).fit([0.001, 0.999] * 10, [0, 1] * 10)
    assert ts.temperature_ == pytest.approx(0.5, rel=1e-3)


def test_sklearn_clone_and_params():
    ts = TemperatureScaling(bounds=(0.1, 20.0), eps=1e-6)
    cloned = clone(ts)
    assert cloned.get_params() == {"bounds": (0.1, 20.0), "eps": 1e-6}


def test_transform_before_fit_raises():
    with pytest.raises(NotFittedError):
        TemperatureScaling().transform(np.array([0.2, 0.8]))


def test_fit_validates_inputs():
    with pytest.raises(ValueError, match="same length"):
        TemperatureScaling().fit([0.2, 0.8], [0])
    with pytest.raises(ValueError, match="range"):
        TemperatureScaling().fit([0.2, 1.5], [0, 1])
    with pytest.raises(ValueError, match="0 and 1"):
        TemperatureScaling().fit([0.2, 0.8], [0, 2])
    with pytest.raises(ValueError, match="empty"):
        TemperatureScaling().fit([], [])


def test_transform_validates_inputs():
    ts = TemperatureScaling().fit([0.2, 0.8, 0.6, 0.4], [0, 1, 1, 0])
    with pytest.raises(ValueError, match="range"):
        ts.transform(np.array([0.2, -0.1]))
    with pytest.raises(ValueError, match="1-dimensional"):
        ts.transform(np.array([[0.2, 0.8]]))


@pytest.mark.parametrize(
    "bounds", [(0.0, 1.0), (-1.0, 5.0), (5.0, 1.0), (1.0, 1.0), 3, ("a", "b")]
)
def test_bad_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        TemperatureScaling(bounds=bounds).fit([0.2, 0.8], [0, 1])


@pytest.mark.parametrize("eps", [0.0, -1e-3, 0.5, 0.7])
def test_bad_eps(eps):
    with pytest.raises(ValueError, match="eps"):
        TemperatureScaling(eps=eps).fit([0.2, 0.8], [0, 1])
