# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend.preprocessing import one_hot


def test_default():
    y = np.array([0, 1, 2, 3, 4, 2])
    expect = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype="float",
    )
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


def test_autoguessing():
    y = np.array([0, 4, 0, 4])
    expect = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype="float",
    )
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


def test_list():
    y = [0, 1, 2, 3, 4, 2]
    expect = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype="float",
    )
    out = one_hot(y)
    np.testing.assert_array_equal(expect, out)


def test_multidim_list():
    y = [[0, 1, 2, 3, 4, 2]]
    with pytest.raises(AttributeError):
        one_hot(y)


def test_multidim_array():
    y = np.array([[0], [1], [2], [3], [4], [2]])
    with pytest.raises(AttributeError):
        one_hot(y)


def test_oneclass():
    np.testing.assert_array_equal(one_hot([0]), np.array([[0.0]], dtype="float"))


def test_list_morelabels():
    y = [0, 1]
    expect = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype="float")
    out = one_hot(y, num_labels=3)
    np.testing.assert_array_equal(expect, out)


def test_single_class_batch_and_empty_explicit_width():
    np.testing.assert_array_equal(one_hot([0, 0, 0]), np.zeros((3, 1)))
    assert one_hot([], num_labels=1).shape == (0, 1)


def test_infer_width_at_uint8_boundary():
    out = one_hot(np.array([0, 255], dtype=np.uint8))
    np.testing.assert_array_equal(out, np.eye(256)[[0, 255]])


def test_tuple_input_and_requested_dtype():
    out = one_hot((0, 1), dtype=bool)
    assert out.dtype == bool
    np.testing.assert_array_equal(out, np.eye(2, dtype=bool))


@pytest.mark.parametrize("labels, width", [([0, -1, 1], "auto"), ([1], 1)])
def test_invalid_label_indices(labels, width):
    with pytest.raises(ValueError):
        one_hot(labels, num_labels=width)
