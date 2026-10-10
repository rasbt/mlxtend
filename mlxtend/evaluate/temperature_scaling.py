# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# A post-hoc probability calibrator for binary classifiers.
# Author: Supriya Selvan Ganesan
#
# License: BSD 3 clause

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .calibration import _check_binary_probs


def _to_logits(y_prob, eps):
    """Convert probabilities to logits, clipping away exact 0 and 1."""
    p = np.clip(y_prob, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


class TemperatureScaling(BaseEstimator):
    """Post-hoc calibration of binary classifier probabilities.

    Many classifiers produce probabilities that are too confident (e.g.,
    Gaussian naive Bayes) or not confident enough. Temperature scaling
    fixes this with a single number, the temperature T: the predicted
    probabilities are converted to logits, divided by T, and converted
    back to probabilities,

    p_calibrated = sigmoid(logit(p) / T)

    T > 1 makes the predictions less confident, T < 1 makes them more
    confident, and T = 1 changes nothing. T is learned by minimizing the
    negative log-likelihood (log loss) on a held-out calibration set.

    Since dividing by a positive T never changes the order of the
    predictions and keeps p = 0.5 at 0.5, the predicted class labels
    (and thus accuracy at a 0.5 threshold) stay exactly the same; only the
    confidence changes.

    Parameters
    -----------
    bounds : tuple of 2 floats (default: (0.01, 100.0))
        Lower and upper limit of the temperature search. Both must be
        positive, and the lower limit must be smaller than the upper one.
    eps : float (default: 1e-12)
        Predicted probabilities are clipped to [eps, 1 - eps] before the
        logit transform to avoid infinite values. Must be in (0, 0.5).

    Attributes
    -----------
    temperature_ : float
        The fitted temperature. If the calibration data are classified
        perfectly with high confidence, the optimum may lie at the lower
        bound of `bounds`.

    Examples
    -----------
    >>> import numpy as np
    >>> from mlxtend.evaluate import TemperatureScaling
    >>> y_prob = np.array([0.01, 0.99, 0.95, 0.05, 0.9, 0.1])
    >>> y_true = np.array([0, 1, 0, 1, 1, 0])
    >>> scaler = TemperatureScaling().fit(y_prob, y_true)
    >>> scaler.temperature_ > 1  # over-confident, so it is softened
    True

    For more usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/TemperatureScaling/
    """

    def __init__(self, bounds=(0.01, 100.0), eps=1e-12):
        self.bounds = bounds
        self.eps = eps

    def _check_params(self):
        try:
            low, high = self.bounds
            low, high = float(low), float(high)
        except (TypeError, ValueError):
            raise ValueError("`bounds` must be a tuple of two positive numbers.")
        if not (0 < low < high):
            raise ValueError("`bounds` must satisfy 0 < lower < upper.")
        if not (0 < self.eps < 0.5):
            raise ValueError("`eps` must be in the range (0, 0.5).")
        return low, high

    def fit(self, y_prob, y_target):
        """Learn the temperature on a held-out calibration set.

        Parameters
        -----------
        y_prob : array-like, shape=[n_samples]
            Predicted probabilities of the positive class.
        y_target : array-like, shape=[n_samples]
            True binary class labels (0 or 1).

        Returns
        -----------
        self : object
        """
        low, high = self._check_params()
        y_target, y_prob = _check_binary_probs(y_target, y_prob)
        logits = _to_logits(y_prob, self.eps)

        def neg_log_likelihood(log_t):
            z = logits / np.exp(log_t)
            # mean of log(1 + exp(z)) - y * z, computed stably
            return float(np.mean(np.logaddexp(0.0, z) - y_target * z))

        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(np.log(low), np.log(high)),
            method="bounded",
            options={"xatol": 1e-8},
        )
        self.temperature_ = float(np.exp(result.x))
        return self

    def transform(self, y_prob):
        """Return temperature-scaled probabilities.

        Parameters
        -----------
        y_prob : array-like, shape=[n_samples]
            Predicted probabilities of the positive class.

        Returns
        -----------
        y_prob_calibrated : array, shape=[n_samples]
            Calibrated probabilities of the positive class.
        """
        check_is_fitted(self, "temperature_")
        self._check_params()
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim != 1:
            raise ValueError("`y_prob` must be 1-dimensional.")
        if np.isnan(y_prob).any() or np.any(y_prob < 0) or np.any(y_prob > 1):
            raise ValueError("`y_prob` values must be in the range [0, 1].")
        return expit(_to_logits(y_prob, self.eps) / self.temperature_)

    def fit_transform(self, y_prob, y_target):
        """Fit the temperature and return the calibrated probabilities."""
        return self.fit(y_prob, y_target).transform(y_prob)
