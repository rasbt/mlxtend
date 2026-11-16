import gc
import importlib
import tracemalloc

import numpy as np
import pytest

from mlxtend.data import (
    boston_housing_data,
    iris_data,
    mnist_data,
    three_blobs_data,
)

DATASETS = [
    ("mnist", mnist_data, {}, (5000, 784), True),
    ("iris", iris_data, {"version": "uci"}, (150, 4), True),
    ("iris", iris_data, {"version": "corrected"}, (150, 4), True),
    ("boston_housing", boston_housing_data, {}, (506, 13), False),
    ("three_blobs", three_blobs_data, {}, (150, 2), True),
]


@pytest.mark.parametrize("module_name,loader,kwargs,shape,integer_labels", DATASETS)
def test_numeric_parser_preserves_every_bundled_value(
    module_name, loader, kwargs, shape, integer_labels
):
    module = importlib.import_module("mlxtend.data." + module_name)
    reference = np.genfromtxt(module.DATA_PATH, delimiter=",")
    assert np.isfinite(reference).all()
    expected_X = reference[:, :-1]
    expected_y = reference[:, -1]
    if kwargs.get("version") == "corrected":
        expected_X[34] = [4.9, 3.1, 1.5, 0.2]
        expected_X[37] = [4.9, 3.6, 1.4, 0.1]
    if integer_labels:
        expected_y = expected_y.astype(int)

    X, y = loader(**kwargs)

    assert X.shape == shape
    assert y.shape == (shape[0],)
    assert X.dtype == expected_X.dtype
    assert y.dtype == expected_y.dtype
    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)


@pytest.mark.parametrize("module_name,loader,kwargs,shape,integer_labels", DATASETS)
def test_loaded_arrays_remain_independent_between_calls(
    module_name, loader, kwargs, shape, integer_labels
):
    first_X, first_y = loader(**kwargs)
    second_X, second_y = loader(**kwargs)
    original_X = second_X.copy()
    original_y = second_y.copy()
    first_X[:] = -9876
    first_y[:] = -9876

    np.testing.assert_array_equal(second_X, original_X)
    np.testing.assert_array_equal(second_y, original_y)
    assert not np.shares_memory(first_X, second_X)
    assert not np.shares_memory(first_y, second_y)


def test_mnist_peak_allocation_stays_close_to_numeric_output_size():
    gc.collect()
    tracemalloc.start()
    try:
        X, y = mnist_data()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # Permit three copies of the numerical result, including parser buffers.
    # The general parser needs over seven copies' worth of traced memory.
    assert peak < 3 * (X.nbytes + y.nbytes), peak


@pytest.mark.parametrize("module_name,loader,kwargs,shape,integer_labels", DATASETS)
@pytest.mark.parametrize("field", ["", "missing", "nan", "inf", "-inf"])
def test_missing_and_nonnumeric_fields_keep_general_parser_results(
    module_name, loader, kwargs, shape, integer_labels, field, tmp_path, monkeypatch
):
    # Also exercise the compatibility path if a bundled CSV gains missing data.
    rows = [[str(i + j / 10.0) for j in range(shape[1] + 1)] for i in range(45)]
    rows[0][0] = field
    path = tmp_path / "values.csv"
    path.write_text("\n".join(",".join(row) for row in rows) + "\n")
    module = importlib.import_module("mlxtend.data." + module_name)
    monkeypatch.setattr(module, "DATA_PATH", str(path))
    reference = np.genfromtxt(path, delimiter=",")
    expected_X, expected_y = reference[:, :-1], reference[:, -1]
    if kwargs.get("version") == "corrected":
        expected_X[34] = [4.9, 3.1, 1.5, 0.2]
        expected_X[37] = [4.9, 3.6, 1.4, 0.1]
    if integer_labels:
        expected_y = expected_y.astype(int)

    X, y = loader(**kwargs)

    np.testing.assert_array_equal(X, expected_X)
    np.testing.assert_array_equal(y, expected_y)
    assert X.dtype == expected_X.dtype
    assert y.dtype == expected_y.dtype


@pytest.mark.parametrize("module_name,loader,kwargs,shape,integer_labels", DATASETS)
def test_empty_files_keep_the_existing_shape_error(
    module_name, loader, kwargs, shape, integer_labels, tmp_path, monkeypatch
):
    path = tmp_path / "empty.csv"
    path.write_text("")
    module = importlib.import_module("mlxtend.data." + module_name)
    monkeypatch.setattr(module, "DATA_PATH", str(path))

    with pytest.warns(UserWarning), pytest.raises(IndexError):
        loader(**kwargs)
