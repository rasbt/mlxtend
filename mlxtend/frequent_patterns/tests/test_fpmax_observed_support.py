import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

from mlxtend.frequent_patterns import fpmax


def _oracle(values, threshold, max_len=None):
    frequent = {}
    for length in range(1, values.shape[1] + 1):
        for columns in itertools.combinations(range(values.shape[1]), length):
            selected = values[:, columns]
            observed = selected[~np.isnan(selected).any(axis=1)]
            if not len(observed):
                continue
            support = (observed == 1).all(axis=1).sum() / len(observed)
            if support >= threshold:
                frequent[frozenset(columns)] = support
    return {
        items: support
        for items, support in frequent.items()
        if not any(items < other for other in frequent)
        and (max_len is None or len(items) <= max_len)
    }


def _check(values, threshold, max_len=None, names=False):
    columns = [f"item-{i}" for i in range(values.shape[1])]
    frame = pd.DataFrame(values, columns=columns, index=np.arange(len(values)) + 17)
    before = frame.copy(deep=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = fpmax(
            frame,
            min_support=threshold,
            null_values=True,
            max_len=max_len,
            use_colnames=names,
        )
    expected = _oracle(values, threshold, max_len)
    if names:
        expected = {
            frozenset(columns[i] for i in items): support
            for items, support in expected.items()
        }
    actual = dict(zip(result.itemsets, result.support))
    assert len(actual) == len(result), "Duplicate maximal itemsets"
    assert actual.keys() == expected.keys()
    for items in expected:
        assert_allclose(actual[items], expected[items], rtol=0, atol=1e-15)
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("names", [False, True])
@pytest.mark.parametrize(
    "values,threshold",
    [
        ([[1, 1], [1, 0], [np.nan, 1]], 0.6),
        ([[1, 1], [0, np.nan], [0, np.nan]], 0.5),
        ([[1, 1, np.nan], [np.nan, 1, 1], [1, np.nan, 1]], 1.0),
        ([[np.nan, np.nan], [0, np.nan], [np.nan, 0]], 0.5),
        (
            [
                [1, 1, 0, 1],
                [0, 0, 0, 1],
                [1, 1, 1, 1],
                [0, np.nan, 0, 1],
                [1, 1, 0, 0],
                [np.nan, np.nan, 1, 1],
            ],
            0.6,
        ),
    ],
)
def test_missing_observations_use_confirmed_maximal_sets(values, threshold, names):
    _check(np.asarray(values, dtype=float), threshold, names=names)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("threshold", [0.2, 0.5, 0.8, 1.0])
@pytest.mark.parametrize("max_len", [None, 1, 2])
def test_missing_itemsets_match_exhaustive_oracle(seed, threshold, max_len):
    values = np.random.RandomState(seed).choice([0.0, 1.0, np.nan], (7, 4))
    _check(values, threshold, max_len)


def test_all_three_transaction_two_item_tables():
    for cells in itertools.product([0.0, 1.0, np.nan], repeat=6):
        values = np.asarray(cells).reshape(3, 2)
        if np.isnan(values).any():
            _check(values, 0.6)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("threshold", [0.07, 0.29, 0.61, 1.0])
def test_complete_rows_provide_only_a_conservative_candidate_bound(seed, threshold):
    values = np.random.RandomState(seed + 40).choice(
        [0.0, 1.0, np.nan], (200, 5), p=[0.25, 0.7, 0.05]
    )
    _check(values, threshold)


def test_length_limit_does_not_make_subsets_maximal():
    values = np.array([[1, 1, 1], [1, 1, 1], [np.nan, 0, 0]], dtype=float)
    for max_len in (1, 2, 3, None):
        _check(values, 0.6, max_len)


def test_verbose_missing_support_path(capsys):
    frame = pd.DataFrame([[1.0, 1.0], [1.0, 0.0], [np.nan, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        fpmax(frame, min_support=0.6, null_values=True, verbose=1)
    assert "itemset(s)" in capsys.readouterr().out
