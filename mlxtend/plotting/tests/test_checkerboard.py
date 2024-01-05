# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import matplotlib.pyplot as plt
import numpy as np

from mlxtend.plotting import checkerboard_plot

plt.switch_backend("agg")


def test_runs():
    ary = np.random.random((6, 4))
    checkerboard_plot(
        ary,
        col_labels=["abc", "def", "ghi", "jkl"],
        row_labels=["sample %d" % i for i in range(1, 6)],
        cell_colors=["skyblue", "whitesmoke"],
        font_colors=["black", "black"],
        figsize=(5, 5),
    )
