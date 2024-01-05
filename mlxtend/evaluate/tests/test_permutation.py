# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend.evaluate import permutation_test
from mlxtend.utils import assert_raises

treatment = [689, 656, 668, 660, 679, 663, 664, 647]
control = [657, 623, 652, 654, 658, 660, 670, 620]


def test_one_sided_x_greater_y():
    p = permutation_test(treatment, control, func=lambda x, y: np.mean(x) - np.mean(y))
    assert round(p, 4) == 0.0301, p

    p = permutation_test(treatment, control, func="x_mean > y_mean")
    assert round(p, 4) == 0.0301, p


def test_one_sided_y_greater_x():
    p = permutation_test(treatment, control, func=lambda x, y: np.mean(y) - np.mean(x))
    assert round(p, 3) == 0.973, p

    p = permutation_test(treatment, control, func="x_mean < y_mean")
    assert round(p, 3) == 0.973, p


def test_two_sided():
    p = permutation_test(
        treatment, control, func=lambda x, y: np.abs(np.mean(x) - np.mean(y))
    )
    assert round(p, 3) == 0.060, p

    p = permutation_test(treatment, control, func="x_mean != y_mean")
    assert round(p, 3) == 0.060, p


def test_default():
    p = permutation_test(treatment, control)
    assert round(p, 3) == 0.060, p


def test_approximateone_sided_x_greater_y():
    p = permutation_test(
        treatment,
        control,
        func=lambda x, y: np.mean(x) - np.mean(y),
        method="approximate",
        num_rounds=5000,
        seed=123,
    )
    assert round(p, 3) == 0.031, p


def test_invalid_method():
    msg = 'method must be "approximate" or "exact", got na'
    assert_raises(
        AttributeError,
        msg,
        permutation_test,
        [1, 2, 3],
        [3, 4, 5],
        lambda x, y: np.mean(x) - np.mean(y),
        method="na",
    )


def test_invalid_func():
    msg = (
        "Provide a custom function lambda x,y: ... "
        'or a string in ("x_mean != y_mean", '
        '"x_mean > y_mean", "x_mean < y_mean")'
    )
    assert_raises(AttributeError, msg, permutation_test, [1, 2, 3], [3, 4, 5], "myfunc")


def test_paired_runs_approximate():
    a = [3.67, 1.72, 3.46, 2.60, 2.03, 2.10, 3.01]
    b = [2.11, 1.79, 2.71, 1.89, 1.69, 1.71, 2.01]
    p = permutation_test(
        a, b, paired=True, method="approximate", seed=0, num_rounds=100000
    )
    assert round(p, 3) == 0.031


def test_paired_runs_exact():
    a = [3.67, 1.72, 3.46, 2.60, 2.03, 2.10, 3.01]
    b = [2.11, 1.79, 2.71, 1.89, 1.69, 1.71, 2.01]
    p = permutation_test(a, b, paired=True, method="exact")
    assert round(p, 3) == 0.031


def test_paired_invalid_lengths():
    msg = "x and y must have the same" " length if `paired=True`"
    assert_raises(ValueError, msg, permutation_test, [1, 2, 3], [3, 4], paired=True)
