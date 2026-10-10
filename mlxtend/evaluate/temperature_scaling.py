# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Temperature scaling to calibrate binary classifier probabilities.
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
    """Turn probabilities into logits, log(p / (1 - p)).

    A probability of exactly 0 or 1 would give infinity, so the values are
    clipped to [eps, 1 - eps] first.
    """
    p = np.clip(y_prob, eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


class TemperatureScaling(BaseEstimator):
    """Fix over- or under-confident probabilities with a single number.

    Some classifiers, Gaussian naive Bayes for example, give probabilities
    that are too extreme. They say 0.99 when 0.8 would be more honest.
    Temperature scaling softens (or sharpens) them using one number, the
    temperature T. Each probability is turned into a logit, divided by T,
    and turned back into a probability:

        p_new = sigmoid(logit(p) / T)

    T > 1 makes the model less confident, T < 1 makes it more confident,
    and T = 1 changes nothing. The best T is the one with the lowest log
    loss on a calibration set. Use data the classifier was not trained on
    for this, otherwise the result will look better than it really is.

    Dividing by a positive T keeps the order of the predictions and keeps
    0.5 at 0.5. That means the predicted classes, and so the accuracy at a
    0.5 threshold, stay exactly the same. Only the confidence changes.

    Parameters
    -----------
    bounds : tuple of 2 floats (default: (0.01, 100.0))
        Range in which T is searched. Both numbers must be positive and the
        first one has to be smaller than the second one.
    eps : float (default: 1e-12)
        Probabilities are clipped to [eps, 1 - eps] before the logit is
        taken, because 0 and 1 would turn into infinity. Must be between
        0 and 0.5.

    Attributes
    -----------
    temperature_ : float
        The fitted temperature. If the calibration data is classified
        perfectly and with high confidence, it can end up at the lower
        end of `bounds`.

    Examples
    -----------
    >>> import numpy as np
    >>> from mlxtend.evaluate import TemperatureScaling
    >>> y_prob = np.array([0.01, 0.99, 0.95, 0.05, 0.9, 0.1])
    >>> y_true = np.array([0, 1, 0, 1, 1, 0])
    >>> scaler = TemperatureScaling().fit(y_prob, y_true)
    >>> scaler.temperature_ > 1  # too confident, so it gets softened
    True

    For more usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/TemperatureScaling/
    """

    def __init__(self, bounds=(0.01, 100.0), eps=1e-12):
        self.bounds = bounds
        self.eps = eps

    def _check_params(self):
        """Validate `bounds` and `eps` and return the bounds as floats."""
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
        """Learn the temperature from a held-out calibration set.

        Parameters
        -----------
        y_prob : array-like, shape=[n_samples]
            Predicted probabilities of the positive class.
        y_target : array-like, shape=[n_samples]
            True labels, either 0 or 1.

        Returns
        -----------
        self : object
        """
        low, high = self._check_params()
        y_target, y_prob = _check_binary_probs(y_target, y_prob)
        logits = _to_logits(y_prob, self.eps)

        def neg_log_likelihood(log_t):
            # we search over log(T) so that T can never be zero or negative
            z = logits / np.exp(log_t)
            # mean of log(1 + exp(z)) - y * z, which is the log loss.
            # logaddexp keeps big values from overflowing.
            return float(np.mean(np.logaddexp(0.0, z) - y_target * z))

        # the loss has one minimum, so a bounded 1D search is enough here
        result = minimize_scalar(
            neg_log_likelihood,
            bounds=(np.log(low), np.log(high)),
            method="bounded",
            options={"xatol": 1e-8},
        )
        self.temperature_ = float(np.exp(result.x))
        return self

    def transform(self, y_prob):
        """Rescale probabilities with the fitted temperature.

        Parameters
        -----------
        y_prob : array-like, shape=[n_samples]
            Predicted probabilities of the positive class.

        Returns
        -----------
        y_prob_calibrated : array, shape=[n_samples]
            The rescaled probabilities of the positive class.
        """
        # raises a NotFittedError if fit was not called yet
        check_is_fitted(self, "temperature_")
        self._check_params()
        y_prob = np.asarray(y_prob, dtype=float)
        if y_prob.ndim != 1:
            raise ValueError("`y_prob` must be 1-dimensional.")
        if np.isnan(y_prob).any() or np.any(y_prob < 0) or np.any(y_prob > 1):
            raise ValueError("`y_prob` values must be in the range [0, 1].")
        return expit(_to_logits(y_prob, self.eps) / self.temperature_)

    def fit_transform(self, y_prob, y_target):
        """Fit the temperature and return the rescaled probabilities."""
        return self.fit(y_prob, y_target).transform(y_prob)
