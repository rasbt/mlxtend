# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os
import sys

import numpy as np

from mlxtend.utils import assert_raises, check_Xy, format_kwarg_dictionaries

y = np.array([1, 2, 3, 4])
X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])

d_default = {"key1": 1, "key2": 2}
d_user = {"key3": 3, "key4": 4}
protected_keys = ["key1", "key4"]


def test_check_Xy_ok():
    check_Xy(X, y)


def test_check_Xy_invalid_type_X():
    expect = "X must be a NumPy array. Found <class 'list'>"
    if sys.version_info < (3, 0):
        expect = expect.replace("class", "type")
    assert_raises(ValueError, expect, check_Xy, [1, 2, 3, 4], y)


def test_check_Xy_float16_X():
    check_Xy(X.astype(np.float16), y)


def test_check_Xy_float16_y():
    check_Xy(X, y.astype(np.int16))


def test_check_Xy_invalid_type_y():
    expect = "y must be a NumPy array. Found <class 'list'>"
    if sys.version_info < (3, 0):
        expect = expect.replace("class", "type")
    assert_raises(ValueError, expect, check_Xy, X, [1, 2, 3, 4])


def test_check_Xy_invalid_dtype_X():
    assert_raises(
        ValueError,
        "X must be an integer or float array. Found object.",
        check_Xy,
        X.astype("object"),
        y,
    )


def test_check_Xy_invalid_dtype_y():
    if sys.version_info > (3, 0):
        expect = (
            "y must be an integer array. Found <U1. "
            "Try passing the array as y.astype(np.int_)"
        )
    else:
        expect = (
            "y must be an integer array. Found |S1. "
            "Try passing the array as y.astype(np.int_)"
        )
    assert_raises(ValueError, expect, check_Xy, X, np.array(["a", "b", "c", "d"]))


def test_check_Xy_invalid_dim_y():
    if sys.version_info[:2] == (2, 7) and os.name == "nt":
        s = "y must be a 1D array. Found (4L, 2L)"
    else:
        s = "y must be a 1D array. Found (4, 2)"
    assert_raises(ValueError, s, check_Xy, X, X.astype(np.int_))


def test_check_Xy_invalid_dim_X():
    if sys.version_info[:2] == (2, 7) and os.name == "nt":
        s = "X must be a 2D array. Found (4L,)"
    else:
        s = "X must be a 2D array. Found (4,)"
    assert_raises(ValueError, s, check_Xy, y, y)


def test_check_Xy_unequal_length_X():
    assert_raises(
        ValueError,
        ("y and X must contain the same number of samples. " "Got y: 4, X: 3"),
        check_Xy,
        X[1:],
        y,
    )


def test_check_Xy_unequal_length_y():
    assert_raises(
        ValueError,
        ("y and X must contain the same number of samples. " "Got y: 3, X: 4"),
        check_Xy,
        X,
        y[1:],
    )


def test_format_kwarg_dictionaries_defaults_empty():
    empty = format_kwarg_dictionaries()
    assert isinstance(empty, dict)
    assert len(empty) == 0


def test_format_kwarg_dictionaries_protected_keys():
    formatted_kwargs = format_kwarg_dictionaries(
        default_kwargs=d_default, user_kwargs=d_user, protected_keys=protected_keys
    )

    for key in protected_keys:
        assert key not in formatted_kwargs


def test_format_kwarg_dictionaries_no_default_kwargs():
    formatted_kwargs = format_kwarg_dictionaries(user_kwargs=d_user)
    assert formatted_kwargs == d_user


def test_format_kwarg_dictionaries_no_user_kwargs():
    formatted_kwargs = format_kwarg_dictionaries(default_kwargs=d_default)
    assert formatted_kwargs == d_default


def test_format_kwarg_dictionaries_default_kwargs_invalid_type():
    invalid_kwargs = "not a dictionary"
    message = "d must be of type dict or None, but got " "{} instead".format(
        type(invalid_kwargs)
    )
    assert_raises(
        TypeError, message, format_kwarg_dictionaries, default_kwargs=invalid_kwargs
    )


def test_format_kwarg_dictionaries_user_kwargs_invalid_type():
    invalid_kwargs = "not a dictionary"
    message = "d must be of type dict or None, but got " "{} instead".format(
        type(invalid_kwargs)
    )
    assert_raises(
        TypeError, message, format_kwarg_dictionaries, user_kwargs=invalid_kwargs
    )
