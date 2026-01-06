import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.svm import SVC

from mlxtend.plotting import plot_decision_regions_3d


def test_plot_decision_regions_3d():
    X = np.array([[1, 2, 3], [4, 5, 6], [1, 1, 1], [5, 5, 5]])
    y = np.array([0, 1, 0, 1])
    clf = SVC().fit(X, y)

    try:
        plot_decision_regions_3d(X, y, clf, z_slices=[1, 3, 5])
        plt.close()
    except Exception as e:
        pytest.fail(f"3D plotting failed: {e}")
