# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# A function for computing the expected calibration error.
# Author: Supriya Selvan Ganesan
#
# License: BSD 3 clause

import numpy as np


def _check_binary_probs(y_target, y_prob):
    """Validate and convert binary targets and positive-class probabilities."""
    y_target = np.asarray(y_target)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_target.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("`y_target` and `y_prob` must be 1-dimensional.")
    if y_target.shape != y_prob.shape:
        raise ValueError("`y_target` and `y_prob` must have the same length.")
    if y_target.size == 0:
        raise ValueError("`y_target` and `y_prob` must not be empty.")
    if not np.isin(y_target, [0, 1]).all():
        raise ValueError("`y_target` must contain only the labels 0 and 1.")
    if np.isnan(y_prob).any() or np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("`y_prob` values must be in the range [0, 1].")
    return y_target, y_prob


def _bin_statistics(y_target, y_prob, n_bins):
    """Group predictions into equal-width bins (low, high].

    Returns
    -------
    counts : array, shape=[n_bins]
        Number of samples per bin.
    mean_prob : array, shape=[n_bins]
        Mean predicted probability per bin (NaN for empty bins).
    frac_pos : array, shape=[n_bins]
        Observed fraction of positives per bin (NaN for empty bins).
    """
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, (int, np.integer))
        or n_bins < 1
    ):
        raise ValueError("`n_bins` must be a positive integer.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # (low, high] bins; a probability of exactly 0.0 falls into the first bin
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    sum_prob = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
    sum_pos = np.bincount(bin_ids, weights=y_target, minlength=n_bins)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prob = np.where(counts > 0, sum_prob / counts, np.nan)
        frac_pos = np.where(counts > 0, sum_pos / counts, np.nan)
    return counts, mean_prob, frac_pos


def expected_calibration_error(y_target, y_prob, n_bins=10):
    """Compute the Expected Calibration Error (ECE) of a binary classifier.

    A classifier is well calibrated if, among all samples to which it assigns
    a positive-class probability of about p, a fraction of about p are
    actually positive. ECE quantifies deviations from this ideal: predictions
    are grouped into equal-width probability bins, and the absolute difference
    between the observed fraction of positives and the mean predicted
    probability in each bin is averaged, weighted by the share of samples
    falling into that bin:

    ECE = sum_b (n_b / n) * | mean(y_b) - mean(p_b) |

    Bins are half-open intervals (low, high]; a probability of exactly 0.0
    falls into the first bin.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True binary class labels (0 or 1).
    y_prob : array-like, shape=[n_samples]
        Predicted probabilities of the positive class, in the range [0, 1].
    n_bins : int (default: 10)
        Number of equal-width probability bins.

    Returns
    ----------
    ece : float
        Expected calibration error in the range [0, 1]. A value of 0 means
        perfectly calibrated; larger values mean worse calibration.

    Examples
    -----------
    >>> import numpy as np
    >>> from mlxtend.evaluate import expected_calibration_error
    >>> y = np.array([0, 1, 1, 1])
    >>> p = np.array([0.1, 0.9, 0.8, 0.3])
    >>> round(expected_calibration_error(y, p, n_bins=2), 3)
    0.225

    For more usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/expected_calibration_error/
    """
    y_target, y_prob = _check_binary_probs(y_target, y_prob)
    counts, mean_prob, frac_pos = _bin_statistics(y_target, y_prob, n_bins)

    nonempty = counts > 0
    weights = counts[nonempty] / counts.sum()
    gaps = np.abs(frac_pos[nonempty] - mean_prob[nonempty])
    return float(np.sum(weights * gaps))
