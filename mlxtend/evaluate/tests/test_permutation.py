# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.evaluate import permutation_test
from mlxtend.utils import assert_raises


treatment = [689, 656, 668, 660, 679, 663, 664, 647]
control = [657, 623, 652, 654, 658, 660, 670, 620]


def test_one_sided():
    p = permutation_test(treatment, control, alt_hypothesis='x > y')
    assert round(p, 4) == 0.0274


def test_numpy_array():
    p = permutation_test(np.array(treatment),
                         np.array(control),
                         alt_hypothesis='x > y')
    assert round(p, 4) == 0.0274


def test_invert_alt_hypothesis():
    p = permutation_test(treatment,
                         control,
                         alt_hypothesis='x < y')
    assert round(p, 2) == 1 - 0.03


def test_two_sided():
    p = permutation_test(treatment,
                         control,
                         alt_hypothesis='x != y')
    assert round(p, 2) == round(2*0.027, 2)


def test_defaults():
    p = permutation_test(treatment,
                         control)
    assert round(p, 2) == round(2*0.027, 2)


def test_approximate():
    p = permutation_test(treatment,
                         control,
                         method='approximate',
                         alt_hypothesis='x > y',
                         num_permutations=5000,
                         seed=123)
    assert round(p, 3) == 0.028, round(p, 4)


def test_invalid_alt_hypothesis():
    msg = "alt_hypothesis be 'x > y', 'y > x', or 'x != y', got y > x"
    assert_raises(AttributeError,
                  msg,
                  permutation_test,
                  [1, 2, 3],
                  [3, 4, 5],
                  alt_hypothesis='y > x')


def test_invalid_method():
    msg = 'method must be "approximate" or "exact", got na'
    assert_raises(AttributeError,
                  msg,
                  permutation_test,
                  [1, 2, 3],
                  [3, 4, 5],
                  alt_hypothesis='x > y',
                  method='na')


def test_invalid_array_dim_x():
    msg = 'x must be one-dimensional'
    assert_raises(AttributeError,
                  msg,
                  permutation_test,
                  np.array([[1, 2, 3]]),
                  [3, 4, 5])


def test_invalid_array_dim_y():
    msg = 'y must be one-dimensional'
    assert_raises(AttributeError,
                  msg,
                  permutation_test,
                  [3, 4, 5],
                  np.array([[1, 2, 3]]),)
