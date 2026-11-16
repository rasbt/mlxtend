# mlxtend Machine Learning Library Extensions
# License: BSD 3 clause

from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix

from mlxtend.frequent_patterns import hmine
from mlxtend.frequent_patterns.hmine import create_projected_itemsets


def _exhaustive_itemsets(values, min_support, max_len, labels):
    expected = {}
    for size in range(1, (max_len or values.shape[1]) + 1):
        for indices in combinations(range(values.shape[1]), size):
            support = np.count_nonzero(values[:, indices].all(axis=1)) / len(values)
            if support >= min_support:
                expected[frozenset(labels[i] for i in indices)] = support
    return expected


@pytest.mark.parametrize("seed", [0, 1, 7, 42])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("max_len", [None, 2, 4])
def test_incremental_projection_matches_exhaustive_supports(seed, sparse, max_len):
    values = np.random.RandomState(seed).random_sample((12, 6)) < 0.65
    before = values.copy()
    frame = (
        pd.DataFrame.sparse.from_spmatrix(csr_matrix(values))
        if sparse
        else pd.DataFrame(values)
    )
    for min_support in [0.25, 0.5, 0.75, 1.0]:
        result = hmine(frame, min_support=min_support, max_len=max_len)
        observed = dict(zip(result["itemsets"], result["support"]))
        assert observed == _exhaustive_itemsets(
            values, min_support, max_len, range(values.shape[1])
        )
        assert len(observed) == len(result)
    assert_array_equal(values, before)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("filled", [False, True])
def test_extreme_density_and_named_items(sparse, filled):
    values = np.full((8, 5), filled, dtype=bool)
    labels = ["one", "two", "three", "four", "five"]
    frame = (
        pd.DataFrame.sparse.from_spmatrix(csr_matrix(values), columns=labels)
        if sparse
        else pd.DataFrame(values, columns=labels)
    )
    result = hmine(frame, min_support=0.5, use_colnames=True)
    assert dict(zip(result["itemsets"], result["support"])) == _exhaustive_itemsets(
        values, 0.5, None, labels
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_projection_helper_still_checks_complete_prefix_on_unconditioned_input(sparse):
    values = np.array(
        [[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 0]], dtype=bool
    )
    data = csr_matrix(values) if sparse else values
    result = create_projected_itemsets([0, 1], data)
    if sparse:
        result = result.toarray()
    assert_array_equal(result, values[[0, 3]])
