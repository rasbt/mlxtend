import gc
import tracemalloc

import numpy as np
import pytest

from mlxtend.feature_extraction import RBFKernelPCA


@pytest.fixture(scope="module", params=[np.float64, np.float32])
def fitted_model(request):
    rng = np.random.RandomState(19)
    training = rng.normal(size=(64, 4)).astype(request.param)
    return RBFKernelPCA(gamma=0.2, n_components=3).fit(training)


def _full_matrix_reference(model, X):
    if len(X) == 0:
        return np.empty((0, model.n_components))
    squared_distances = np.array(
        [((model.X_ - point) ** 2).sum(axis=1) for point in X], dtype=float
    )
    kernel = np.exp(-model.gamma * squared_distances)
    components = model.e_vecs_[:, : model.n_components]
    return kernel @ (components / model.e_vals_[: model.n_components])


@pytest.mark.parametrize("n_query", [0, 1, 8191, 8192, 8193])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_block_projection_matches_full_matrix_at_boundaries(
    fitted_model, n_query, layout
):
    model = fitted_model
    X = np.random.RandomState(8).normal(size=(n_query, 4)).astype(model.X_.dtype)
    if layout == "F":
        X = np.asfortranarray(X)
    elif layout == "strided":
        storage = np.zeros((n_query, 8), dtype=X.dtype)
        storage[:, ::2] = X
        X = storage[:, ::2]
    original = X.copy()
    training = model.X_.copy()
    expected = _full_matrix_reference(model, X)

    actual = model.transform(X)

    tolerance = 32 * np.finfo(X.dtype).eps
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert actual.shape == (n_query, 3)
    np.testing.assert_array_equal(X, original)
    np.testing.assert_array_equal(model.X_, training)


def test_query_order_and_separate_calls_do_not_change_projection(fitted_model):
    X = np.random.RandomState(22).normal(size=(9003, 4)).astype(fitted_model.X_.dtype)
    whole = fitted_model.transform(X)
    separately = np.concatenate(
        [fitted_model.transform(X[:17]), fitted_model.transform(X[17:])]
    )
    np.testing.assert_allclose(whole, separately, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        fitted_model.transform(X[::-1]), whole[::-1], rtol=1e-12, atol=1e-14
    )


def test_transform_peak_memory_is_bounded_for_large_query_batches():
    rng = np.random.RandomState(41)
    model = RBFKernelPCA(gamma=0.2, n_components=3).fit(rng.normal(size=(256, 4)))
    X = rng.normal(size=(12000, 4))
    gc.collect()
    tracemalloc.start()
    try:
        actual = model.transform(X)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert actual.shape == (12000, 3)
    assert peak < 8 * 1024 * 1024, peak
