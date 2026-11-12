# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.naive_bayes import GaussianNB

from mlxtend.classifier import EnsembleVoteClassifier


def _dataset(labels):
    X = np.concatenate([np.linspace(-0.5, 0.5, 6) + k * 3 for k in range(3)])
    return X.reshape(-1, 1), np.repeat(labels, 6)


@pytest.mark.parametrize("labels", [[10, 20, 30], ["cat", "dog", "fox"]])
@pytest.mark.parametrize(
    "subsets", [([0, 1], [1, 2]), ([0], [0, 1, 2]), ([0, 1, 2], [0, 1, 2])]
)
def test_prefit_soft_voting_aligns_each_estimators_class_columns(labels, subsets):
    labels = np.asarray(labels)
    X, y = _dataset(labels)
    classifiers = []
    for subset in subsets:
        rows = np.isin(y, labels[list(subset)])
        classifiers.append(GaussianNB().fit(X[rows], y[rows]))
    original_means = [clf.theta_.copy() for clf in classifiers]
    weights = [1, 3]
    ensemble = EnsembleVoteClassifier(
        clfs=classifiers,
        voting="soft",
        weights=weights,
        use_clones=False,
        fit_base_estimators=False,
    ).fit(X, y)

    expected = np.zeros((len(X), len(labels)))
    for weight, clf in zip(weights, classifiers):
        probabilities = clf.predict_proba(X)
        for j, label in enumerate(labels):
            if label in clf.classes_:
                column = list(clf.classes_).index(label)
                expected[:, j] += weight * probabilities[:, column] / sum(weights)

    assert_allclose(ensemble.predict_proba(X), expected)
    assert_allclose(ensemble.predict_proba(X).sum(axis=1), 1)
    assert_array_equal(ensemble.predict(X), labels[expected.argmax(axis=1)])
    assert ensemble.transform(X).shape == (2, len(X), len(labels))
    for clf, original in zip(classifiers, original_means):
        assert_array_equal(clf.theta_, original)


def test_prefit_soft_voting_respects_a_different_learned_class_order():
    X, y = _dataset(np.array([10, 20, 30]))
    classifier = GaussianNB().fit(X, y)
    expected = classifier.predict_proba(X)
    permutation = [2, 0, 1]
    for attribute in ["classes_", "class_count_", "class_prior_", "theta_", "var_"]:
        setattr(classifier, attribute, getattr(classifier, attribute)[permutation])

    ensemble = EnsembleVoteClassifier(
        clfs=[classifier], voting="soft", use_clones=False, fit_base_estimators=False
    ).fit(X, y)

    assert_allclose(ensemble.predict_proba(X), expected)
    assert_array_equal(ensemble.predict(X), classifier.predict(X))


def test_prefit_hard_voting_retains_label_predictions():
    X, y = _dataset(np.array([10, 20, 30]))
    classifier = GaussianNB().fit(X, y)
    ensemble = EnsembleVoteClassifier(
        clfs=[classifier], voting="hard", use_clones=False, fit_base_estimators=False
    ).fit(X, y)

    assert_array_equal(ensemble.predict(X), classifier.predict(X))
