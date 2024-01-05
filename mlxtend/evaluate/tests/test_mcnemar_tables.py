# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend.evaluate import mcnemar_tables
from mlxtend.utils import assert_raises


def test_input_array_1d():
    t = np.array([[1, 2], [3, 4]])
    assert_raises(
        ValueError,
        "One or more input arrays are not 1-dimensional.",
        mcnemar_tables,
        t,
        t,
        t,
    )


def test_input_array_lengths_1():
    t = np.array([1, 2])
    t2 = np.array([1, 2, 3])
    assert_raises(
        ValueError,
        ("Each prediction array must have" " the same number of samples."),
        mcnemar_tables,
        t,
        t2,
        t,
    )


def test_model_have_same_len():
    y_true = np.array([1, 1, 0])
    y_1 = np.array([0, 1, 0])
    y_2 = np.array([1, 1, 0])
    y_3 = np.array([1, 1])

    assert_raises(
        ValueError,
        ("Each prediction array must have" " the same number of samples."),
        mcnemar_tables,
        y_true,
        y_1,
        y_2,
        y_3,
    )


def test_min_number_of_models():
    y_true = np.array([1, 1, 0])
    y_1 = np.array([0, 1, 0])

    assert_raises(
        ValueError,
        ("Provide at least 2 model prediction arrays."),
        mcnemar_tables,
        y_true,
        y_1,
    )


def test_input_binary_all_right():
    y_target = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tb = mcnemar_tables(y_target, y_model1, y_model2)
    expect = np.array([[8, 0], [0, 0]])

    assert len(tb) == 1
    np.testing.assert_array_equal(tb["model_0 vs model_1"], expect)


def test_input_binary_all_wrong():
    y_target = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_model1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_model2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    tb = mcnemar_tables(y_target, y_model1, y_model2)
    expect = np.array([[0, 0], [0, 8]])

    assert len(tb) == 1
    np.testing.assert_array_equal(tb["model_0 vs model_1"], expect)


def test_input_binary():
    y_target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_model1 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    y_model2 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    y_model3 = np.array([0, 1, 1, 0, 0, 1, 1, 0, 0, 0])
    tb = mcnemar_tables(y_target, y_model1, y_model2, y_model3)

    expect1 = np.array([[4, 2], [1, 3]])

    expect2 = np.array([[5, 1], [0, 4]])

    expect3 = np.array([[4, 1], [1, 4]])

    assert len(tb) == 3
    np.testing.assert_array_equal(tb["model_0 vs model_1"], expect1)
    np.testing.assert_array_equal(tb["model_0 vs model_2"], expect2)
    np.testing.assert_array_equal(tb["model_1 vs model_2"], expect3)


def test_input_nonbinary():
    y_target = np.array([0, 0, 0, 0, 0, 2, 2, 3, 3, 3])
    y_model1 = np.array([0, 1, 0, 0, 0, 2, 2, 1, 1, 1])
    y_model2 = np.array([0, 0, 1, 1, 0, 2, 2, 0, 1, 0])
    y_model3 = np.array([0, 1, 1, 0, 0, 2, 2, 0, 1, 1])
    tb = mcnemar_tables(y_target, y_model1, y_model2, y_model3)

    expect1 = np.array([[4, 2], [1, 3]])

    expect2 = np.array([[5, 1], [0, 4]])

    expect3 = np.array([[4, 1], [1, 4]])

    assert len(tb) == 3
    np.testing.assert_array_equal(tb["model_0 vs model_1"], expect1)
    np.testing.assert_array_equal(tb["model_0 vs model_2"], expect2)
    np.testing.assert_array_equal(tb["model_1 vs model_2"], expect3)
