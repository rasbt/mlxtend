"""Tests for the Iris model comparison example."""

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


EXAMPLE_PATH = (
    Path(__file__).parents[1]
    / "examples"
    / "plot_iris_model_comparison.py"
)

SPEC = importlib.util.spec_from_file_location(
    "plot_iris_model_comparison",
    EXAMPLE_PATH,
)

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_classifiers_returns_three_classifiers():
    """The example should create three individual classifiers."""
    classifiers = MODULE.build_classifiers()

    assert len(classifiers) == 3


def test_evaluate_classifiers_returns_valid_scores():
    """Each classifier should produce an accuracy between zero and one."""
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.30,
        random_state=42,
        stratify=iris.target,
    )

    classifiers = MODULE.build_classifiers()

    results = MODULE.evaluate_classifiers(
        classifiers,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    assert len(results) == 3
    assert all(0.0 <= score <= 1.0 for score in results.values())
    assert all(np.isfinite(score) for score in results.values())


def test_plot_results_creates_output_file(tmp_path):
    """The plotting function should save a non-empty image file."""
    results = {
        "Model A": 0.90,
        "Model B": 0.95,
        "Model C": 0.93,
    }

    output_path = tmp_path / "test_plot.png"

    MODULE.plot_results(results, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0