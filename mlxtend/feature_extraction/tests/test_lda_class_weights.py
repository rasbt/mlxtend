from itertools import permutations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh

from mlxtend.feature_extraction import LinearDiscriminantAnalysis


def _data(counts):
    labels = np.repeat(np.arange(3), counts)
    centers = np.array([[0.0, 0.0], [3.0, 1.0], [-1.0, 4.0]])
    samples = np.random.RandomState(11).normal(size=(sum(counts), 2))
    return samples + centers[labels], labels


@pytest.mark.parametrize("counts", [(3, 5, 7), (5, 5, 5), (7, 3, 5)])
def test_lda_uses_matching_class_counts_in_scatter(counts):
    X, y = _data(counts)
    model = LinearDiscriminantAnalysis(n_discriminants=2).fit(X, y)

    # Solve the independent generalized scatter problem without forming W^-1 B.
    between = np.zeros((2, 2))
    within = np.zeros((2, 2))
    for label in np.unique(y):
        samples = X[y == label]
        mean = samples.mean(axis=0)
        delta = mean - X.mean(axis=0)
        between += len(samples) * np.outer(delta, delta)
        centered = samples - mean
        within += centered.T @ centered
    values, vectors = eigh(between, within)
    values, vectors = values[::-1], vectors[:, ::-1]
    vectors /= np.linalg.norm(vectors, axis=0)
    signs = np.sign(np.sum(vectors * model.e_vecs_, axis=0))

    assert_allclose(model.e_vals_, values, rtol=1e-12, atol=1e-12)
    assert_allclose(model.e_vecs_, vectors * signs, rtol=1e-12, atol=1e-12)
    assert_allclose(
        model.transform(X), X @ (vectors * signs), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("label_order", list(permutations(range(3))))
def test_lda_is_invariant_to_class_label_permutations(label_order):
    X, y = _data((3, 5, 7))
    reference = LinearDiscriminantAnalysis(n_discriminants=2).fit(X, y)
    relabeled = np.asarray(label_order)[y]
    model = LinearDiscriminantAnalysis(n_discriminants=2).fit(X, relabeled)

    assert_allclose(model.e_vals_, reference.e_vals_, rtol=1e-12, atol=1e-12)
    assert_allclose(
        np.abs(model.e_vecs_), np.abs(reference.e_vecs_), rtol=1e-12, atol=1e-12
    )
