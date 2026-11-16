import numpy as np


def _load_numeric_csv(path):
    """Read numeric datasets, retaining the missing-value parser fallback."""
    try:
        return np.loadtxt(path, delimiter=",")
    except ValueError:
        return np.genfromtxt(path, delimiter=",")
