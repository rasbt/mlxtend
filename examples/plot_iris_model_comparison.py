"""
Compare individual classifiers with a soft-voting ensemble on the Iris dataset.

This example demonstrates how mlxtend's EnsembleVoteClassifier can combine
different machine-learning models into one voting ensemble.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlxtend.classifier import EnsembleVoteClassifier


def build_classifiers():
    """Create the individual classifiers used in the comparison."""
    logistic_regression = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    decision_tree = DecisionTreeClassifier(
        max_depth=3,
        random_state=42,
    )

    random_forest = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
    )

    return [logistic_regression, decision_tree, random_forest]


def evaluate_classifiers(classifiers, X_train, X_test, y_train, y_test):
    """Train each classifier and return its test accuracy."""
    results = {}

    for classifier in classifiers:
        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)
        name = classifier[-1].__class__.__name__ if isinstance(
            classifier, Pipeline
        ) else classifier.__class__.__name__
        results[name] = accuracy_score(y_test, predictions)

    return results


def plot_results(results, output_path):
    """Create and save a bar chart of classifier accuracy scores."""
    names = list(results.keys())
    scores = list(results.values())

    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(names, scores, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])

    axis.set_title("Iris Classification: Individual Models vs Voting Ensemble")
    axis.set_xlabel("Classifier")
    axis.set_ylabel("Test Accuracy")
    axis.set_ylim(0.80, 1.01)
    axis.grid(axis="y", linestyle="--", alpha=0.4)

    for bar, score in zip(bars, scores):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.005,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main():
    """Run the Iris classifier comparison."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.30,
        random_state=42,
        stratify=iris.target,
    )

    classifiers = build_classifiers()

    results = evaluate_classifiers(
        classifiers,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    voting_classifier = EnsembleVoteClassifier(
        clfs=build_classifiers(),
        voting="soft",
    )

    voting_classifier.fit(X_train, y_train)
    voting_predictions = voting_classifier.predict(X_test)
    results["EnsembleVoteClassifier"] = accuracy_score(
        y_test,
        voting_predictions,
    )

    for name, score in results.items():
        print(f"{name}: accuracy={score:.3f}")

    output_path = Path(__file__).with_name("iris_model_comparison.png")
    plot_results(results, output_path)
    print(f"Plot saved to: {output_path}")


if __name__ == "__main__":
    main()