# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Expected calibration error for binary classifiers.
# Author: Supriya Selvan Ganesan
#
# License: BSD 3 clause

import numpy as np


def _check_binary_probs(y_target, y_prob):
    """Check the inputs and turn them into numpy arrays.

    Both inputs have to be 1D and have the same length. The labels must be
    0 or 1 and the probabilities must be between 0 and 1, otherwise a
    ValueError is raised.
    """
    y_target = np.asarray(y_target)
    y_prob = np.asarray(y_prob, dtype=float)

    # we only handle flat lists of numbers, not 2D arrays
    if y_target.ndim != 1 or y_prob.ndim != 1:
        raise ValueError("`y_target` and `y_prob` must be 1-dimensional.")
    if y_target.shape != y_prob.shape:
        raise ValueError("`y_target` and `y_prob` must have the same length.")
    if y_target.size == 0:
        raise ValueError("`y_target` and `y_prob` must not be empty.")
    if not np.isin(y_target, [0, 1]).all():
        raise ValueError("`y_target` must contain only the labels 0 and 1.")
    # NaN fails every comparison, so it needs its own check
    if np.isnan(y_prob).any() or np.any(y_prob < 0) or np.any(y_prob > 1):
        raise ValueError("`y_prob` values must be in the range [0, 1].")
    return y_target, y_prob


def _bin_statistics(y_target, y_prob, n_bins):
    """Sort the predictions into equal-width bins and summarize each bin.

    Returns
    -------
    counts : array, shape=[n_bins]
        How many samples ended up in each bin.
    mean_prob : array, shape=[n_bins]
        Average predicted probability in each bin (NaN if the bin is empty).
    frac_pos : array, shape=[n_bins]
        Fraction of samples in each bin that are really positive
        (NaN if the bin is empty).
    """
    # True is technically an int in Python, so rule it out explicitly
    if (
        isinstance(n_bins, bool)
        or not isinstance(n_bins, (int, np.integer))
        or n_bins < 1
    ):
        raise ValueError("`n_bins` must be a positive integer.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize only needs the inner edges, 0 and 1 are the outer limits.
    # With right=True the bins are (low, high], so a probability of exactly
    # 0.0 lands in the first bin and 1.0 in the last one.
    bin_ids = np.digitize(y_prob, edges[1:-1], right=True)

    # add up what is in each bin, then divide by the count to get averages
    counts = np.bincount(bin_ids, minlength=n_bins).astype(float)
    sum_prob = np.bincount(bin_ids, weights=y_prob, minlength=n_bins)
    sum_pos = np.bincount(bin_ids, weights=y_target, minlength=n_bins)

    # empty bins give 0/0, so mark them NaN instead of crashing or warning
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_prob = np.where(counts > 0, sum_prob / counts, np.nan)
        frac_pos = np.where(counts > 0, sum_pos / counts, np.nan)
    return counts, mean_prob, frac_pos


def expected_calibration_error(y_target, y_prob, n_bins=10):
    """Compute the expected calibration error (ECE) of a binary classifier.

    A model is well calibrated when its probabilities match what really
    happens. If it says 0.8 for a bunch of samples, about 80% of them
    should turn out positive. ECE tells you how far off the model is.

    The predictions are sorted into equal-width bins by predicted
    probability. In each bin we compare the average predicted probability
    with the fraction of samples that are actually positive. ECE is the
    average of those gaps, where bins with more samples count more:

        ECE = sum over bins of (n_bin / n) * |fraction_positive - mean_prob|

    The bins are (low, high], so a probability of exactly 0.0 goes into the
    first bin.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True labels, either 0 or 1.
    y_prob : array-like, shape=[n_samples]
        Predicted probability of the positive class, between 0 and 1.
    n_bins : int (default: 10)
        Number of equal-width bins to use.

    Returns
    ----------
    ece : float
        A number between 0 and 1. 0 means perfectly calibrated, and bigger
        numbers mean worse calibration.

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

    # empty bins have no mean, so they are left out
    nonempty = counts > 0
    # a bin counts in proportion to how many samples it holds
    weights = counts[nonempty] / counts.sum()
    # gap between what the model said and what actually happened
    gaps = np.abs(frac_pos[nonempty] - mean_prob[nonempty])
    return float(np.sum(weights * gaps))
