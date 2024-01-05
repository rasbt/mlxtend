# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np
import pytest

from mlxtend.plotting import heatmap

plt.switch_backend("agg")


def test_defaults():
    heatmap(np.random.random((10, 5)))


def test_wrong_column_name_number():
    with pytest.raises(AssertionError) as excinfo:
        heatmap(np.random.random((10, 5)), column_names=["a", "b", "c"])
        assert excinfo.value.message == (
            "len(column_names) (got 3)"
            " should be equal to number of"
            " rows in the input "
            " array (expect 5)."
        )


def test_wrong_row_name_number():
    with pytest.raises(AssertionError) as excinfo:
        heatmap(np.random.random((10, 5)), row_names=["a", "b", "c"])
        assert excinfo.value.message == (
            "len(column_names) (got 3)"
            " should be equal to number of"
            " rows in the input "
            " array (expect 10)."
        )
