# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _BaseClassifier
import numpy as np
from mlxtend.utils import assert_raises


def test_init():
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    assert hasattr(cl, 'print_progress')
    assert hasattr(cl, 'random_seed')


def test_check_labels_ok_1():
    y = np.array([1, 1, 0])
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    cl._check_target_array(y=y, allowed={(0, 1)})


def test_check_labels_ok_2():
    y = np.array([1, 2, 2])
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    cl._check_target_array(y=y, allowed={(1, 2), (0, 1)})


def test_check_labels_not_ok_1():
    y = np.array([1, 3, 2])
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    assert_raises(AttributeError,
                  'Labels not in {(1, 2), (0, 1)}.\nFound (1, 2, 3)',
                  cl._check_target_array,
                  y,
                  {(0, 1), (1, 2)})


def test_check_labels_interger_notok():
    y = np.array([1., 2.], dtype=np.float64)
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    assert_raises(AttributeError,
                  'y must be an integer array.\nFound float64',
                  cl._check_target_array,
                  y)


def test_check_labels_positive_notok():
    y = np.array([1, 1, -1])
    cl = _BaseClassifier(print_progress=0, random_seed=1)
    assert_raises(AttributeError,
                  'y array must not contain negative labels.\nFound [-1  1]',
                  cl._check_target_array,
                  y)
