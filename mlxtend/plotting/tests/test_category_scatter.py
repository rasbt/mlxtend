# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from mlxtend.plotting import category_scatter


@pytest.mark.parametrize("as_frame", [False, True])
@pytest.mark.parametrize("legend_loc", ["best", False])
def test_existing_axes(as_frame, legend_loc):
    data = np.array([[1, 2, 0], [3, 4, 1], [5, 6, 0]])
    columns = [0, 1, 2]
    if as_frame:
        columns = ["x", "y", "label"]
        data = pd.DataFrame(data, columns=columns)
    fig, axes = plt.subplots(1, 2)
    other = plt.figure()
    try:
        figures = plt.get_fignums()
        for ax in axes:
            result = category_scatter(*columns, data, ax=ax, legend_loc=legend_loc)
            assert result is fig
            assert len(ax.collections) == 2
            np.testing.assert_array_equal(
                ax.collections[0].get_offsets(), [[1, 2], [5, 6]]
            )
            np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[3, 4]])
            assert (ax.get_legend() is not None) == bool(legend_loc)
        assert plt.get_fignums() == figures
        assert not other.axes
    finally:
        plt.close(fig)
        plt.close(other)


def test_default_creates_new_figure():
    existing = plt.figure()
    result = None
    try:
        result = category_scatter(0, 1, 2, np.array([[1, 2, 0], [3, 4, 1]]))
        assert result is not existing
        assert len(result.axes) == 1
        assert len(result.axes[0].collections) == 2
        assert not existing.axes
    finally:
        plt.close(existing)
        if result is not None:
            plt.close(result)
