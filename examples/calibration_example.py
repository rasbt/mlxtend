"""Compare how well-calibrated three classifiers are.

Run from the repository root:
    python examples/calibration_example.py

A classifier can have high accuracy but still output misleading
probabilities (e.g. saying "95% sure" when it is right only 70% of the
time). This script trains three models on the same data and reports
accuracy, Expected Calibration Error (ECE) and Maximum Calibration
Error (MCE), then prints the reliability table for the worst model.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mlxtend.evaluate import (
    expected_calibration_error,
    maximum_calibration_error,
    reliability_table,
)


def main():
    # Load a real binary dataset and keep a held-out test set.
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000)
        ),
        "Gaussian Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }

    print(f"{'Model':<22}{'Accuracy':>10}{'ECE':>8}{'MCE':>8}")
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)
        acc = (proba.argmax(axis=1) == y_test).mean()
        ece = expected_calibration_error(y_test, proba, n_bins=10)
        mce = maximum_calibration_error(y_test, proba, n_bins=10)
        results[name] = (ece, proba)
        print(f"{name:<22}{acc:>10.3f}{ece:>8.3f}{mce:>8.3f}")

    # Show where the least calibrated model goes wrong.
    worst = max(results, key=lambda k: results[k][0])
    table = reliability_table(y_test, results[worst][1], n_bins=10)
    print(f"\nReliability table for {worst}:")
    print(f"{'bin':<14}{'count':>6}{'confidence':>12}{'accuracy':>10}")
    for lo, hi, n, conf, acc in zip(table["bin_lower"], table["bin_upper"],
                                    table["count"], table["avg_confidence"],
                                    table["accuracy"]):
        print(f"({lo:.1f}, {hi:.1f}]{n:>10}{conf:>12.3f}{acc:>10.3f}")


if __name__ == "__main__":
    main()
