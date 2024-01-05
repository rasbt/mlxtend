# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend.evaluate import mcnemar_table
from mlxtend.utils import assert_raises


def test_input_array_1d():
    t = np.array([[1, 2], [3, 4]])
    assert_raises(
        ValueError,
        "One or more input arrays are not 1-dimensional.",
        mcnemar_table,
        t,
        t,
        t,
    )


def test_input_array_lengths_1():
    t = np.array([1, 2])
    t2 = np.array([1, 2, 3])
    assert_raises(
        ValueError,
        ("y_target and y_model1 contain a different number" " of elements."),
        mcnemar_table,
        t,
        t2,
        t,
    )


def test_input_array_lengths_2():
    t = np.array([1, 2])
    t2 = np.array([1, 2, 3])
    assert_raises(
        ValueError,
        ("y_target and y_model2 contain a different number" " of elements."),
        mcnemar_table,
        t,
        t,
        t2,
    )


def test_input_binary_all_right():
    y_target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tb = mcnemar_table(y_target=y_target, y_model1=y_model1, y_model2=y_model2)
    expect = np.array([[8, 0], [0, 0]])
    np.testing.assert_array_equal(tb, expect)


def test_input_binary_all_wrong():
    y_target = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_model1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tb = mcnemar_table(y_target=y_target, y_model1=y_model1, y_model2=y_model2)
    expect = np.array([[0, 0], [0, 8]])

    np.testing.assert_array_equal(tb, expect)


def test_input_binary():
    y_target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_model1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    y_model2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    tb = mcnemar_table(y_target=y_target, y_model1=y_model1, y_model2=y_model2)
    expect = np.array([[4, 2], [1, 3]])

    np.testing.assert_array_equal(tb, expect)


def test_input_nonbinary():
    y_target = np.array([0, 0, 0, 0, 0, 2, 1, 1, 1, 1])
    y_model1 = np.array([0, 5, 0, 0, 0, 2, 1, 0, 0, 0])
    y_model2 = np.array([0, 0, 1, 3, 0, 2, 1, 0, 0, 0])

    tb = mcnemar_table(y_target=y_target, y_model1=y_model1, y_model2=y_model2)
    expect = np.array([[4, 2], [1, 3]])

    np.testing.assert_array_equal(tb, expect)
