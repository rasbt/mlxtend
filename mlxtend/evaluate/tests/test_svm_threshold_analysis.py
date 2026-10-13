
"""Tests for the SVM threshold analysis example."""

import importlib.util
from pathlib import Path

import pytest

EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "svm_threshold_analysis.py"
)

spec = importlib.util.spec_from_file_location("svm_threshold_analysis", EXAMPLE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

evaluate_thresholds = module.evaluate_thresholds


def test_perfect_predictions():
    results = evaluate_thresholds([0, 1, 0, 1], [-2, 2, -1, 1], [0])
    assert results[0]["precision"] == 1.0
    assert results[0]["recall"] == 1.0
    assert results[0]["f1"] == 1.0


def test_no_positive_predictions():
    results = evaluate_thresholds([0, 1], [-2, -1], [0])
    assert results[0]["precision"] == 0.0
    assert results[0]["recall"] == 0.0
    assert results[0]["f1"] == 0.0


def test_multiple_thresholds():
    results = evaluate_thresholds([0, 1], [-1, 1], [-2, 0, 2])
    assert len(results) == 3
    assert results[1]["f1"] == pytest.approx(1.0)
