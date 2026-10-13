import numpy as np


def zscore_outlier_mask(X, threshold=3.0):
    """Return a boolean mask marking rows that are not z-score outliers.

    A row is kept only if every feature value lies within ``threshold``
    standard deviations of its column mean.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
        Training vectors.
    threshold : float, default=3.0
        Maximum absolute z-score allowed for each value.

    Returns
    -------
    mask : numpy.ndarray, shape = [n_samples]
        Boolean array; ``True`` marks rows to keep.

    Raises
    ------
    ValueError
        If ``X`` is empty or ``threshold`` is not positive.

    Examples
    --------
    >>> import numpy as np
    >>> from mlxtend.preprocessing import zscore_outlier_mask
    >>> X = np.array([[1.0], [2.0], [3.0], [100.0]])
    >>> zscore_outlier_mask(X, threshold=1.5)
    array([ True,  True,  True, False])
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        raise ValueError("X must contain at least one sample.")
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    std = X.std(axis=0)
    std[std == 0] = 1.0
    z = np.abs((X - X.mean(axis=0)) / std)
    return np.all(z <= threshold, axis=1)
