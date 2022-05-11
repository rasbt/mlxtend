# mlxtend Machine Learning Library Extensions
#
# Time series cross validation with grouping.
# Author: Dmitry Labazkin <labdmitriy@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score

from mlxtend.evaluate import GroupTimeSeriesSplit


@pytest.fixture
def X():
    return np.array(
        [[0], [7], [6], [4], [4], [8], [0], [6], [2], [0], [5], [9], [7], [7], [7], [7]]
    )


@pytest.fixture
def y():
    return np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0])


@pytest.fixture
def group_numbers():
    return np.array([0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5])


@pytest.fixture
def not_sorted_group_numbers():
    return np.array([5, 5, 5, 5, 1, 1, 1, 1, 3, 3, 2, 2, 2, 4, 4, 0])


@pytest.fixture
def not_consecutive_group_numbers():
    return np.array([0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 2, 2, 2, 2])


@pytest.fixture
def group_names():
    return np.array(
        [
            "2021-01",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-03",
            "2021-03",
            "2021-03",
            "2021-04",
            "2021-04",
            "2021-05",
            "2021-05",
            "2021-06",
            "2021-06",
            "2021-06",
            "2021-06",
        ]
    )


@pytest.fixture
def not_sorted_group_names():
    return np.array(
        [
            "2021-06",
            "2021-06",
            "2021-06",
            "2021-06",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-04",
            "2021-04",
            "2021-03",
            "2021-03",
            "2021-03",
            "2021-05",
            "2021-05",
            "2021-01",
        ]
    )


@pytest.fixture
def not_consecutive_group_names():
    return np.array(
        [
            "2021-01",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-02",
            "2021-03",
            "2021-03",
            "2021-03",
            "2021-04",
            "2021-04",
            "2021-05",
            "2021-05",
            "2021-03",
            "2021-03",
            "2021-03",
            "2021-03",
        ]
    )


def check_splits(X, y, groups, cv_args, expected_results):
    cv = GroupTimeSeriesSplit(**cv_args)
    results = list(cv.split(X, y, groups))

    assert len(results) == len(expected_results)

    for split, expected_split in zip(results, expected_results):
        assert np.array_equal(split[0], expected_split[0])
        assert np.array_equal(split[1], expected_split[1])

    return cv


def test_get_n_splits(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),
        (np.array([5, 6, 7, 8, 9, 10, 11]), np.array([12, 13, 14, 15])),
    ]
    cv = check_splits(X, y, group_numbers, cv_args, expected_results)

    assert cv.get_n_splits() == len(expected_results)


def test_train_size(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),
        (np.array([5, 6, 7, 8, 9, 10, 11]), np.array([12, 13, 14, 15])),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_train_size_group_names(X, y, group_names):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7]), np.array([8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11])),
        (np.array([5, 6, 7, 8, 9, 10, 11]), np.array([12, 13, 14, 15])),
    ]
    check_splits(X, y, group_names, cv_args, expected_results)


def test_n_splits(X, y, group_numbers):
    cv_args = {"test_size": 2, "n_splits": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11])),
        (np.array([5, 6, 7, 8, 9]), np.array([10, 11, 12, 13, 14, 15])),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_n_splits_gap_size(X, y, group_numbers):
    cv_args = {"test_size": 1, "n_splits": 3, "gap_size": 1}
    expected_results = [
        (np.array([0, 1, 2, 3, 4]), np.array([8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7]), np.array([10, 11])),
        (np.array([5, 6, 7, 8, 9]), np.array([12, 13, 14, 15])),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_n_splits_shift_size(X, y, group_numbers):
    cv_args = {"test_size": 1, "n_splits": 3, "gap_size": 1}
    expected_results = [
        (np.array([0, 1, 2, 3, 4]), np.array([8, 9])),
        (np.array([1, 2, 3, 4, 5, 6, 7]), np.array([10, 11])),
        (np.array([5, 6, 7, 8, 9]), np.array([12, 13, 14, 15])),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_n_splits_expanding_window(X, y, group_numbers):
    cv_args = {"test_size": 3, "n_splits": 3, "window_type": "expanding"}
    expected_results = [
        (np.array([0]), np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])),
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9, 10, 11])),
        (
            np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            np.array([8, 9, 10, 11, 12, 13, 14, 15]),
        ),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_full_usage_of_data(X, y, group_numbers):
    cv_args = {"test_size": 3, "train_size": 2, "n_splits": 2}
    expected_results = [
        (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9, 10, 11])),
        (
            np.array([1, 2, 3, 4, 5, 6, 7]),
            np.array([8, 9, 10, 11, 12, 13, 14, 15]),
        ),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_partial_usage_of_data(X, y, group_numbers):
    cv_args = {"test_size": 2, "train_size": 2, "n_splits": 2}
    expected_results = [
        (np.array([1, 2, 3, 4, 5, 6, 7]), np.array([8, 9, 10, 11])),
        (np.array([5, 6, 7, 8, 9]), np.array([10, 11, 12, 13, 14, 15])),
    ]
    check_splits(X, y, group_numbers, cv_args, expected_results)


def test_not_sorted_group_numbers(X, y, not_sorted_group_numbers):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11, 12])),
        (np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]), np.array([13, 14])),
        (np.array([8, 9, 10, 11, 12, 13, 14]), np.array([15])),
    ]

    check_splits(X, y, not_sorted_group_numbers, cv_args, expected_results)


def test_not_sorted_group_names(X, y, not_sorted_group_names):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = [
        (np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), np.array([10, 11, 12])),
        (np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]), np.array([13, 14])),
        (np.array([8, 9, 10, 11, 12, 13, 14]), np.array([15])),
    ]

    check_splits(X, y, not_sorted_group_names, cv_args, expected_results)


def test_not_specified_train_size_n_splits(X, y, group_numbers):
    cv_args = {"test_size": 1}
    expected_results = None
    error_message = "Either train_size or n_splits should be defined"

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_bad_window_type(X, y, group_numbers):
    cv_args = {
        "test_size": 1,
        "train_size": 3,
        "window_type": "incorrect_window_type",
    }
    expected_results = None
    error_message = 'Window type can be either "rolling" or "expanding"'

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_train_size_with_expanding_window(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 3, "window_type": "expanding"}
    expected_results = None
    error_message = "Train size can be specified only with rolling window"

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_not_specified_groups(X, y):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = None
    error_message = "The groups should be specified"

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, None, cv_args, expected_results)


def test_not_consecutive_group_numbers(X, y, not_consecutive_group_numbers):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = None
    error_message = "The groups should be consecutive"

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, not_consecutive_group_numbers, cv_args, expected_results)


def test_not_consecutive_group_names(X, y, not_consecutive_group_names):
    cv_args = {"test_size": 1, "train_size": 3}
    expected_results = None
    error_message = "The groups should be consecutive"

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, not_consecutive_group_names, cv_args, expected_results)


def test_too_large_train_size_(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 10}
    expected_results = None
    error_message = (
        r"Not enough data to split number of groups \(6\)"
        r" for number splits \(-4\) with train size \(10\),"
        r" test size \(1\), gap size \(0\), shift size \(1\)"
    )

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_too_large_n_splits(X, y, group_numbers):
    cv_args = {"test_size": 1, "n_splits": 10}
    expected_results = None
    error_message = (
        r"Not enough data to split number of groups \(6\)"
        r" for number splits \(10\) with train size \(-4\),"
        r" test size \(1\), gap size \(0\), shift size \(1\)"
    )

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_too_large_train_size_n_splits(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 10, "n_splits": 10}
    expected_results = None
    error_message = (
        r"Not enough data to split number of groups \(6\)"
        r" for number splits \(10\) with train size \(10\),"
        r" test size \(1\), gap size \(0\), shift size \(1\)"
    )

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_too_large_shift_size(X, y, group_numbers):
    cv_args = {"test_size": 1, "n_splits": 3, "shift_size": 10}
    expected_results = None
    error_message = (
        r"Not enough data to split number of groups \(6\)"
        r" for number splits \(3\) with train size \(-15\),"
        r" test size \(1\), gap size \(0\), shift size \(10\)"
    )

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_too_large_gap_size(X, y, group_numbers):
    cv_args = {"test_size": 1, "n_splits": 3, "gap_size": 10}
    expected_results = None
    error_message = (
        r"Not enough data to split number of groups \(6\)"
        r" for number splits \(3\) with train size \(-7\),"
        r" test size \(1\), gap size \(10\), shift size \(1\)"
    )

    with pytest.raises(ValueError, match=error_message):
        check_splits(X, y, group_numbers, cv_args, expected_results)


def test_cross_val_score(X, y, group_numbers):
    cv_args = {"test_size": 1, "train_size": 3}
    cv = GroupTimeSeriesSplit(**cv_args)

    expected_scores = np.array([0, 0.5, 0.25])
    clf = DummyClassifier(strategy="most_frequent")
    scoring = "accuracy"
    cv_scores = cross_val_score(clf, X, y, groups=group_numbers, scoring=scoring, cv=cv)

    assert np.array_equal(cv_scores, expected_scores)
