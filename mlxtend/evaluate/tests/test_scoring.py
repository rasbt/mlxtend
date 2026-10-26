# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pytest

from mlxtend.evaluate import scoring


def test_metric_argument():
    "Test exception is raised when user provides invalid metric argument"
    try:
        scoring(y_target=[1], y_predicted=[1], metric="test")
        assert False
    except AttributeError:
        assert True


def test_y_arguments():
    "Test exception is raised when user provides invalid vectors"
    try:
        scoring(y_target=[1, 2], y_predicted=[1])
        assert False
    except AttributeError:
        assert True


def test_accuracy():
    "Test accuracy metric"
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="accuracy")
    assert res == 0.75


def test_error():
    "Test error metric"
    y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
    y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="error")
    assert res == 0.25


def test_binary():
    "Test exception is raised if label is not binary in f1"
    try:
        y_targ = [1, 1, 1, 0, 0, 2, 0, 3]
        y_pred = [1, 0, 1, 0, 0, 2, 1, 3]
        scoring(y_target=y_targ, y_predicted=y_pred, metric="f1")
        assert False
    except AttributeError:
        assert True


def test_precision():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="precision")
    assert round(res, 3) == 0.75, res


def test_recall():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="recall")
    assert round(res, 3) == 0.6, res


def test_truepositiverate():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="true_positive_rate")
    assert round(res, 3) == 0.6, res


def test_falsepositiverate():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="false_positive_rate")
    assert round(res, 3) == 0.333, res


def test_specificity():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="specificity")
    assert round(res, 3) == 0.667, res


def test_sensitivity():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="sensitivity")
    assert round(res, 3) == 0.6, res


def test_f1():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="f1")
    assert round(res, 3) == 0.667, res


def test_matthews_corr_coef():
    y_targ = [1, 1, 1, 0, 0, 1, 0, 1]
    y_pred = [1, 0, 1, 0, 0, 0, 1, 1]
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="matthews_corr_coef")
    assert round(res, 3) == 0.258, res


def test_balanced_accuracy():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="balanced accuracy")
    assert round(res, 3) == 0.578, res


def test_avg_perclass_accuracy():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(
        y_target=y_targ, y_predicted=y_pred, metric="average per-class accuracy"
    )
    assert round(res, 3) == 0.667, res


def test_avg_perclass_error():
    y_targ = np.array([0, 0, 0, 1, 1, 1, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 0, 1, 1, 2, 2, 2, 2])
    res = scoring(y_target=y_targ, y_predicted=y_pred, metric="average per-class error")
    assert round(res, 3) == 0.333, res


@pytest.mark.parametrize("labels", [[0, 1, 2], [0, 1], ["auto", "other", "third"]])
def test_explicit_numpy_class_vocabulary_matches_list(labels):
    if isinstance(labels[0], str):
        y = np.array(["auto", "auto", "other", "other", "third", "third"])
        pred = np.array(["auto", "other", "other", "other", "third", "auto"])
    else:
        y = np.array([0, 0, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 1, 2, 0])
    expected = scoring(y, pred, metric="average per-class error", unique_labels=labels)
    actual = scoring(
        y, pred, metric="average per-class error", unique_labels=np.array(labels)
    )
    assert actual == pytest.approx(expected)


def test_auto_and_list_controls():
    y = [0, 0, 1, 1, 2, 2]
    pred = [0, 1, 1, 1, 2, 0]
    for labels in ("auto", [0, 1, 2], (0, 1, 2)):
        assert scoring(
            y, pred, metric="average per-class error", unique_labels=labels
        ) == pytest.approx(2 / 9)


def test_binary_scoring_with_numpy_class_vocabulary():
    assert scoring(
        [0, 0, 1, 1], [0, 1, 1, 1], metric="precision", unique_labels=np.array([0, 1])
    ) == pytest.approx(2 / 3)
