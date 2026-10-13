# mlxtend Machine Learning Library Extensions
#
# Calibration metrics for probabilistic classifiers:
# Expected Calibration Error (ECE), Maximum Calibration Error (MCE)
# and the underlying reliability-diagram table.
#
# License: BSD 3 clause

import numpy as np


def _validate_inputs(y_target, y_proba, n_bins, strategy):
    """Check and convert inputs; return (y_true, confidence, correct).

    For binary problems ``y_proba`` may be 1D (probability of class 1)
    or 2D with shape [n_samples, 2]. For multiclass problems it must be
    2D with shape [n_samples, n_classes]. Confidence is the probability
    of the predicted class, and ``correct`` is 1 where the predicted
    class equals the true class.
    """
    y_true = np.asarray(y_target)
    proba = np.asarray(y_proba, dtype=float)

    if y_true.ndim != 1:
        raise ValueError("y_target must be a 1D array.")
    if proba.shape[0] != y_true.shape[0]:
        raise ValueError(
            "y_target and y_proba must have the same number of samples, "
            "got %d and %d." % (y_true.shape[0], proba.shape[0])
        )
    if y_true.shape[0] == 0:
        raise ValueError("Inputs must contain at least one sample.")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer.")
    if strategy not in ("uniform", "quantile"):
        raise ValueError("strategy must be 'uniform' or 'quantile'.")
    if np.any(np.isnan(proba)):
        raise ValueError("y_proba contains NaN values.")
    if np.any(proba < 0) or np.any(proba > 1):
        raise ValueError("y_proba values must be in the range [0, 1].")

    if proba.ndim == 1:
        # Binary case: proba is P(y == 1); labels must be 0/1.
        if not np.all(np.isin(y_true, [0, 1])):
            raise ValueError(
                "With 1D y_proba, y_target must contain only 0 and 1."
            )
        predicted = (proba >= 0.5).astype(int)
        confidence = np.where(predicted == 1, proba, 1.0 - proba)
    elif proba.ndim == 2:
        if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("Each row of a 2D y_proba must sum to 1.")
        if np.any(y_true < 0) or np.any(y_true >= proba.shape[1]):
            raise ValueError(
                "y_target labels must be integers in [0, n_classes)."
            )
        predicted = np.argmax(proba, axis=1)
        confidence = proba[np.arange(proba.shape[0]), predicted]
    else:
        raise ValueError("y_proba must be a 1D or 2D array.")

    correct = (predicted == y_true).astype(float)
    return y_true, confidence, correct


def _bin_edges(confidence, n_bins, strategy):
    """Return bin edges over [0, 1] for the chosen binning strategy."""
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    # Quantile binning: roughly equal number of samples per bin.
    edges = np.quantile(confidence, np.linspace(0.0, 1.0, n_bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    return np.unique(edges)


def reliability_table(y_target, y_proba, n_bins=10, strategy="uniform"):
    """Compute per-bin statistics used for a reliability diagram.

    Samples are grouped by the confidence of their predicted class.
    For each bin, the table reports how confident the model was on
    average and how often it was actually right.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels (integers starting at 0).
    y_proba : array-like, shape=[n_samples] or [n_samples, n_classes]
        Predicted probabilities, e.g. from ``clf.predict_proba(X)``.
        A 1D array is interpreted as the probability of class 1.
    n_bins : int (default: 10)
        Number of confidence bins.
    strategy : str (default: 'uniform')
        'uniform' splits [0, 1] into equal-width bins; 'quantile' uses
        bins holding roughly the same number of samples.

    Returns
    ----------
    table : dict of numpy arrays
        Keys: 'bin_lower', 'bin_upper', 'count', 'avg_confidence',
        'accuracy', 'gap' (= |accuracy - avg_confidence|).
        Empty bins are dropped.

    Examples
    -----------
    >>> from mlxtend.evaluate import reliability_table
    >>> t = reliability_table([0, 1, 1], [0.2, 0.9, 0.6], n_bins=4)
    >>> t['count'].tolist()
    [1, 2]
    """
    _, confidence, correct = _validate_inputs(
        y_target, y_proba, n_bins, strategy
    )
    edges = _bin_edges(confidence, n_bins, strategy)

    # np.digitize with right=True puts a value equal to an edge in the
    # lower bin; clip so that confidence == 0 lands in the first bin.
    idx = np.clip(np.digitize(confidence, edges[1:-1], right=True),
                  0, len(edges) - 2)

    rows = {k: [] for k in ("bin_lower", "bin_upper", "count",
                            "avg_confidence", "accuracy", "gap")}
    for b in range(len(edges) - 1):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        conf = float(confidence[mask].mean())
        acc = float(correct[mask].mean())
        rows["bin_lower"].append(edges[b])
        rows["bin_upper"].append(edges[b + 1])
        rows["count"].append(n)
        rows["avg_confidence"].append(conf)
        rows["accuracy"].append(acc)
        rows["gap"].append(abs(acc - conf))

    table = {k: np.asarray(v) for k, v in rows.items()}
    table["count"] = table["count"].astype(int)
    return table


def expected_calibration_error(y_target, y_proba, n_bins=10,
                               strategy="uniform"):
    """Expected Calibration Error (ECE) of a probabilistic classifier.

    A model is well calibrated when, among predictions made with
    confidence p, about a fraction p are correct. ECE is the
    sample-weighted average gap between confidence and accuracy:

        ECE = sum_b (n_b / N) * |acc(b) - conf(b)|

    0 means perfectly calibrated; larger values mean the predicted
    probabilities are less trustworthy (over- or under-confident).

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels (integers starting at 0).
    y_proba : array-like, shape=[n_samples] or [n_samples, n_classes]
        Predicted probabilities. A 1D array is P(class 1).
    n_bins : int (default: 10)
        Number of confidence bins.
    strategy : str (default: 'uniform')
        'uniform' or 'quantile' binning.

    Returns
    ----------
    ece : float
        Expected Calibration Error in [0, 1].

    Examples
    -----------
    >>> from mlxtend.evaluate import expected_calibration_error
    >>> round(expected_calibration_error([1, 1], [0.9, 0.9]), 2)
    0.1
    """
    table = reliability_table(y_target, y_proba, n_bins, strategy)
    weights = table["count"] / table["count"].sum()
    return float(np.sum(weights * table["gap"]))


def maximum_calibration_error(y_target, y_proba, n_bins=10,
                              strategy="uniform"):
    """Maximum Calibration Error (MCE): the worst gap over all bins.

    Useful in high-risk settings where the worst-case miscalibration
    matters more than the average.

    Parameters
    -----------
    y_target, y_proba, n_bins, strategy :
        Same as in ``expected_calibration_error``.

    Returns
    ----------
    mce : float
        Maximum Calibration Error in [0, 1].
    """
    table = reliability_table(y_target, y_proba, n_bins, strategy)
    return float(np.max(table["gap"]))
