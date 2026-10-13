"""Compare SVM decision thresholds on a binary classification task."""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def evaluate_thresholds(y_true, scores, thresholds=None):
    """Return precision, recall, and F1 for each decision threshold."""
    if thresholds is None:
        thresholds = [-1.0, -0.5, 0.0, 0.5, 1.0]

    results = []
    for threshold in thresholds:
        predictions = (np.asarray(scores) >= threshold).astype(int)
        results.append(
            {
                "threshold": threshold,
                "precision": precision_score(y_true, predictions, zero_division=0),
                "recall": recall_score(y_true, predictions, zero_division=0),
                "f1": f1_score(y_true, predictions, zero_division=0),
            }
        )
    return results


def main():
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        weights=[0.8, 0.2],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = make_pipeline(StandardScaler(), SVC(kernel="linear"))
    model.fit(X_train, y_train)

    scores = model.decision_function(X_test)
    results = evaluate_thresholds(y_test, scores)

    print("Threshold  Precision  Recall  F1")
    for row in results:
        print(
            f"{row['threshold']:8.2f}  "
            f"{row['precision']:9.3f}  "
            f"{row['recall']:6.3f}  {row['f1']:.3f}"
        )


if __name__ == "__main__":
    main()
