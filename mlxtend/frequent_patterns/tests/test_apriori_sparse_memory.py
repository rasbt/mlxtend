# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy import sparse

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns.apriori import generate_new_combinations_low_memory

SPARSE_TYPES = (
    sparse.csr_matrix,
    sparse.csc_matrix,
    sparse.csr_array,
    sparse.csc_array,
)


def _reject_dense_conversion(*args, **kwargs):
    raise AssertionError("Sparse support counting must not densify candidate blocks")


def _without_dense_conversion(monkeypatch):
    for sparse_type in SPARSE_TYPES:
        monkeypatch.setattr(sparse_type, "toarray", _reject_dense_conversion)


def _transactions(dtype):
    return np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=dtype,
    )


def _counts(values, width, min_support):
    result = []
    for itemset in combinations(range(values.shape[1]), width):
        count = np.count_nonzero(np.all(values[:, itemset], axis=1))
        if count / values.shape[0] >= min_support:
            result.append((count, *itemset))
    return result


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
@pytest.mark.parametrize("dtype", [bool, np.uint8])
def test_sparse_candidates_match_enumerated_supports(sparse_type, dtype, monkeypatch):
    values = _transactions(dtype)
    matrix = sparse_type(values)
    _without_dense_conversion(monkeypatch)

    for width in (2, 3, 4):
        previous = np.array(list(combinations(range(values.shape[1]), width - 1)))
        for min_support in (1 / 8, 3 / 8, 1.0):
            result = np.fromiter(
                generate_new_combinations_low_memory(
                    previous, matrix, min_support, True
                ),
                dtype=int,
            ).reshape(-1, width + 1)
            expected = np.array(_counts(values, width, min_support), dtype=int).reshape(
                -1, width + 1
            )
            assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [bool, np.uint8])
@pytest.mark.parametrize("max_len", [None, 2])
def test_sparse_apriori_matches_brute_force(dtype, max_len, monkeypatch):
    values = _transactions(dtype)
    names = ["a", "b", "c", "d", "e"]
    frame = pd.DataFrame.sparse.from_spmatrix(sparse.csc_matrix(values), columns=names)
    _without_dense_conversion(monkeypatch)

    for min_support in (1 / 8, 3 / 8, 1.0):
        result = apriori(
            frame,
            min_support=min_support,
            use_colnames=True,
            low_memory=True,
            max_len=max_len,
        )
        expected = [
            (count / len(values), frozenset(names[i] for i in items))
            for width in range(1, (max_len or len(names)) + 1)
            for count, *items in _counts(values, width, min_support)
        ]
        assert list(result.itertuples(index=False, name=None)) == expected


@pytest.mark.parametrize("sparse_type", SPARSE_TYPES)
def test_explicit_sparse_zeros_do_not_satisfy_a_prefix(sparse_type, monkeypatch):
    matrix = sparse_type(
        (
            np.array([1, 0, 1, 1, 0, 1, 1], dtype=np.uint8),
            (
                np.array([0, 1, 2, 0, 1, 2, 3]),
                np.array([0, 0, 0, 1, 1, 1, 1]),
            ),
        ),
        shape=(4, 2),
    )
    assert matrix.nnz == 7
    _without_dense_conversion(monkeypatch)
    assert list(
        generate_new_combinations_low_memory(np.array([[0], [1]]), matrix, 0.5, True)
    ) == [2, 0, 1]


def test_sparse_prefix_count_does_not_overflow_uint8(monkeypatch):
    values = sparse.csc_matrix(np.ones((2, 257), dtype=np.uint8))
    previous = np.array(list(combinations(range(257), 256)))
    _without_dense_conversion(monkeypatch)
    assert list(generate_new_combinations_low_memory(previous, values, 1.0, True)) == [
        2,
        *range(257),
    ]


@pytest.mark.parametrize("sparse_type", [sparse.csc_matrix, sparse.csc_array])
def test_duplicate_boolean_entries_keep_dense_logical_values(sparse_type, monkeypatch):
    matrix = sparse_type(
        (
            np.ones(6, dtype=bool),
            np.array([0, 0, 1, 0, 0, 1]),
            np.array([0, 3, 6]),
        ),
        shape=(2, 2),
    )
    original_data = matrix.data.copy()
    original_indices = matrix.indices.copy()
    original_indptr = matrix.indptr.copy()
    assert not matrix.has_canonical_format
    _without_dense_conversion(monkeypatch)
    assert list(
        generate_new_combinations_low_memory(np.array([[0], [1]]), matrix, 1.0, True)
    ) == [2, 0, 1]
    assert_array_equal(matrix.data, original_data)
    assert_array_equal(matrix.indices, original_indices)
    assert_array_equal(matrix.indptr, original_indptr)


def test_sparse_prefix_without_matching_rows(monkeypatch):
    matrix = sparse.csc_matrix(np.eye(3, dtype=bool))
    _without_dense_conversion(monkeypatch)
    assert (
        list(
            generate_new_combinations_low_memory(
                np.array([[0, 1], [0, 2], [1, 2]]), matrix, 1 / 3, True
            )
        )
        == []
    )


def test_dense_candidate_path_is_unchanged():
    values = _transactions(bool)
    previous = np.array([[i] for i in range(values.shape[1])])
    result = np.fromiter(
        generate_new_combinations_low_memory(previous, values, 3 / 8, False),
        dtype=int,
    ).reshape(-1, 3)
    assert_array_equal(result, np.array(_counts(values, 2, 3 / 8)))
