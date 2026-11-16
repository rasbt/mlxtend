import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import LogisticRegression

from mlxtend.feature_extraction import LinearDiscriminantAnalysis


@pytest.mark.parametrize(
    "seed,n_samples,n_features,n_classes",
    [(2, 12, 4, 2), (11, 18, 5, 3), (7, 24, 6, 3), (19, 24, 6, 4)],
)
@pytest.mark.parametrize("keep_all", [False, True])
def test_real_training_data_produces_real_usable_projections(
    seed, n_samples, n_features, n_classes, keep_all
):
    X = (
        np.random.RandomState(seed)
        .randint(-4, 5, size=(n_samples, n_features))
        .astype(float)
    )
    y = np.repeat(np.arange(n_classes), n_samples // n_classes)
    original = X.copy()
    model = LinearDiscriminantAnalysis(
        n_discriminants=None if keep_all else n_classes - 1
    ).fit(X, y)
    projected = model.transform(X)

    assert np.isrealobj(model.e_vals_)
    assert np.isrealobj(model.e_vecs_)
    assert np.isrealobj(projected)
    assert np.all(np.isfinite(projected))
    assert projected.shape == (n_samples, n_features if keep_all else n_classes - 1)
    assert LogisticRegression().fit(projected, y).predict(projected).shape == y.shape
    assert_array_equal(X, original)
    assert_allclose(np.linalg.norm(model.e_vecs_, axis=0), 1.0, atol=1e-14)


@pytest.mark.parametrize("n_features,rank", [(4, 1), (5, 2), (6, 3), (4, 4)])
@pytest.mark.parametrize("scale", [1e-100, 1.0, 1e100])
def test_generalized_eigenpairs_match_an_independent_congruence_oracle(
    n_features, rank, scale
):
    rng = np.random.RandomState(31 + rank)
    lower = np.tril(rng.normal(size=(n_features, n_features)))
    lower[np.diag_indices(n_features)] = np.linspace(1.0, 2.0, n_features)
    rotation, _ = np.linalg.qr(rng.normal(size=(n_features, n_features)))
    expected_values = np.r_[np.arange(rank, 0, -1), np.zeros(n_features - rank)]
    within = scale * lower.dot(lower.T)
    between = scale * (lower.dot(rotation) * expected_values).dot(lower.dot(rotation).T)
    expected_vectors = np.linalg.solve(lower.T, rotation)
    expected_vectors /= np.linalg.norm(expected_vectors, axis=0)
    within_before, between_before = within.copy(), between.copy()
    within.flags.writeable = False
    between.flags.writeable = False

    values, vectors = LinearDiscriminantAnalysis()._eigendecom(within, between)

    assert np.isrealobj(values)
    assert np.isrealobj(vectors)
    assert_allclose(values, expected_values, atol=1e-12, rtol=1e-12)
    assert_allclose(np.linalg.norm(vectors, axis=0), 1.0, atol=1e-14)
    assert_allclose(
        (between / scale).dot(vectors),
        (within / scale).dot(vectors) * values,
        atol=1e-11,
        rtol=1e-11,
    )
    # Nonzero eigenvalues are distinct; outer products remove arbitrary signs.
    for index in range(rank):
        assert_allclose(
            np.outer(vectors[:, index], vectors[:, index]),
            np.outer(expected_vectors[:, index], expected_vectors[:, index]),
            atol=1e-11,
            rtol=1e-11,
        )
    assert_array_equal(within, within_before)
    assert_array_equal(between, between_before)


def test_zero_between_scatter_has_a_real_normalized_null_basis():
    within = np.array([[2.0, 0.5], [0.5, 1.0]])
    values, vectors = LinearDiscriminantAnalysis()._eigendecom(
        within, np.zeros_like(within)
    )
    assert np.isrealobj(vectors)
    assert_array_equal(values, np.zeros(2))
    assert_allclose(np.linalg.norm(vectors, axis=0), 1.0)


def test_singular_within_scatter_still_raises():
    with pytest.raises(np.linalg.LinAlgError):
        LinearDiscriminantAnalysis()._eigendecom(np.diag([1.0, 0.0]), np.eye(2))
