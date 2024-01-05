# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend._base import _BaseModel, _Classifier


class BlankClassifier(_BaseModel, _Classifier):
    def __init__(self, print_progress=0, random_seed=1):
        self.print_progress = print_progress
        self.random_seed = random_seed

    def _fit(self, X, y, init_params=True):
        pass

    def _predict(self, X):
        pass


def test_init():
    cl = BlankClassifier(print_progress=0, random_seed=1)
    assert hasattr(cl, "print_progress")
    assert hasattr(cl, "random_seed")


def test_check_labels_ok_1():
    y = np.array([1, 1, 0])
    cl = BlankClassifier(print_progress=0, random_seed=1)
    cl._check_target_array(y=y, allowed={(0, 1)})


def test_check_labels_ok_2():
    y = np.array([1, 2, 2])
    cl = BlankClassifier(print_progress=0, random_seed=1)
    cl._check_target_array(y=y, allowed={(1, 2), (0, 1)})


def test_check_labels_not_ok_1():
    y = np.array([1, 3, 2])
    cl = BlankClassifier(print_progress=0, random_seed=1)
    with pytest.raises(AttributeError) as excinfo:
        cl._check_target_array(y, {(0, 1), (1, 2)})
        assert excinfo.value.message == (
            "Labels not in" " {(1, 2), (0, 1)}." "\nFound (1, 2, 3)"
        )


def test_check_labels_integer_notok():
    y = np.array([1.0, 2.0], dtype=np.float_)
    cl = BlankClassifier(print_progress=0, random_seed=1)
    with pytest.raises(AttributeError) as excinfo:
        cl._check_target_array(y)
        assert excinfo.value.message == (
            "y must be an integer" " array.\nFound float64"
        )


def test_check_labels_positive_notok():
    y = np.array([1, 1, -1])
    cl = BlankClassifier(print_progress=0, random_seed=1)
    with pytest.raises(AttributeError) as excinfo:
        cl._check_target_array(y)
        assert excinfo.value.message == (
            "y array must not " "contain negative " "labels.\nFound [-1  1]"
        )


def test_predict_fail():
    X = np.array([[1], [2], [3]])
    est = BlankClassifier(print_progress=0, random_seed=1)
    est._is_fitted = False
    with pytest.raises(AttributeError) as excinfo:
        est.predict(X)
        assert excinfo.value.message == ("Model is not " "fitted, yet.")


def test_predict_pass():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    est = BlankClassifier(print_progress=0, random_seed=1)
    est.fit(X, y)
    est.predict(X)


def test_fit_1():
    X = np.array([[1], [2], [3]])
    est = BlankClassifier(print_progress=0, random_seed=1)
    with pytest.raises(TypeError) as excinfo:
        est.fit(X)
        assert excinfo.value.message == (
            "fit() missing 1" "required positional argument: 'y'"
        )


def test_fit_2():
    X = np.array([[1], [2], [3]])
    y = np.array([1, 2, 3])
    est = BlankClassifier(print_progress=0, random_seed=1)
    est.fit(X=X, y=y)
