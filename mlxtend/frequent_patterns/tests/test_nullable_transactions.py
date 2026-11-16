import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from mlxtend.frequent_patterns import (
    apriori,
    association_rules,
    fpgrowth,
    fpmax,
    hmine,
)

DTYPES = ["boolean", "Int64", "Float64", object]


def _transactions(dtype, values=None):
    if values is None:
        values = [[1, 1], [1, 0], [pd.NA, 1]]
    return pd.DataFrame(values, columns=["a", "b"], index=[7, 3, 9], dtype=dtype)


def _supports(result):
    return dict(zip(result["itemsets"], result["support"]))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
@pytest.mark.parametrize("use_colnames", [False, True])
def test_nullable_miners_match_observed_transaction_support(dtype, miner, use_colnames):
    data = _transactions(dtype)
    original = data.copy(deep=True)
    a, b = ("a", "b") if use_colnames else (0, 1)
    expected = {frozenset([a, b]): 0.5}
    if miner is fpgrowth:
        expected.update({frozenset([a]): 1.0, frozenset([b]): 2.0 / 3.0})

    actual = miner(data, min_support=0.25, null_values=True, use_colnames=use_colnames)

    assert _supports(actual) == expected
    assert_frame_equal(data, original)


@pytest.mark.parametrize("dtype", DTYPES)
def test_nullable_rule_metrics_match_equivalent_numeric_input(dtype):
    data = _transactions(dtype)
    original = data.copy(deep=True)
    numeric = _transactions(float, [[1, 1], [1, 0], [np.nan, 1]])
    itemsets = pd.DataFrame(
        {
            "support": [1.0, 2.0 / 3.0, 0.5],
            "itemsets": [frozenset(["a"]), frozenset(["b"]), frozenset(["a", "b"])],
        }
    )
    kwargs = dict(num_itemsets=3, null_values=True, min_threshold=0)
    expected = association_rules(itemsets, df_orig=numeric, **kwargs)
    actual = association_rules(itemsets, df_orig=data, **kwargs)

    assert_frame_equal(actual, expected)
    assert len(actual) == 2
    np.testing.assert_allclose(actual["support"], 0.5)
    assert_frame_equal(data, original)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
def test_nullable_mixed_columns_preserve_float_reference_results(dtype, miner):
    numeric = pd.DataFrame(
        [[1, 1, 0], [1, 0, 1], [np.nan, 1, 1], [0, np.nan, 1], [1, 1, 1]],
        columns=["a", "b", "c"],
    )
    data = numeric.copy()
    data["a"] = pd.Series([1, 1, pd.NA, 0, 1], dtype=dtype)
    original = data.copy(deep=True)
    kwargs = dict(min_support=0.1, null_values=True, use_colnames=True)
    expected = _supports(miner(numeric, **kwargs))
    actual = _supports(miner(data, **kwargs))

    assert actual == expected
    assert_frame_equal(data, original)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
def test_all_missing_nullable_transactions_match_empty_numeric_result(dtype, miner):
    data = _transactions(dtype, [[pd.NA, pd.NA]] * 3)
    original = data.copy(deep=True)
    with np.errstate(invalid="ignore"):
        actual = miner(data, min_support=0.25, null_values=True)

    assert actual.empty
    assert list(actual.columns) == ["support", "itemsets"]
    assert_frame_equal(data, original)


@pytest.mark.parametrize("miner", [apriori, fpgrowth, fpmax, hmine])
def test_nullable_boolean_missing_values_are_rejected_when_disabled(miner):
    with pytest.raises(ValueError, match="NaN values are not permitted"):
        miner(_transactions("boolean"), min_support=0.25)


def test_rule_input_rejects_nullable_boolean_missing_values_when_disabled():
    itemsets = pd.DataFrame({"support": [1.0], "itemsets": [frozenset(["a"])]})
    with pytest.raises(ValueError, match="NaN values are not permitted"):
        association_rules(itemsets, df_orig=_transactions("boolean"))


@pytest.mark.parametrize("invalid", [2, "1", "invalid"])
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
def test_missing_normalization_does_not_coerce_invalid_observed_values(invalid, miner):
    data = _transactions(object, [[1, 1], [invalid, 0], [pd.NA, 1]])
    with pytest.raises(ValueError, match="The allowed values"):
        miner(data, min_support=0.25, null_values=True)


@pytest.mark.parametrize("invalid", [2, "1", "invalid"])
def test_rule_input_validates_observed_values_before_normalization(invalid):
    data = _transactions(object, [[1, 1], [invalid, 0], [pd.NA, 1]])
    itemsets = pd.DataFrame({"support": [1.0], "itemsets": [frozenset(["a"])]})
    with pytest.raises(ValueError, match="The allowed values"):
        association_rules(itemsets, num_itemsets=3, df_orig=data, null_values=True)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
def test_nullable_inputs_without_missing_values_keep_existing_supports(dtype, miner):
    data = _transactions(dtype, [[1, 1], [1, 0], [0, 1]])
    expected = miner(data.astype(bool), min_support=0.25, use_colnames=True)
    actual = miner(data, min_support=0.25, use_colnames=True)
    assert _supports(actual) == _supports(expected)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("miner", [fpgrowth, fpmax])
def test_ordinary_dense_and_sparse_boolean_inputs_keep_existing_results(sparse, miner):
    data = pd.DataFrame([[True, True], [True, False], [False, True]])
    if sparse:
        data = data.astype(pd.SparseDtype(bool, False))
    original = data.copy(deep=True)
    actual = miner(data, min_support=0.25)
    expected = {frozenset([0, 1]): 1.0 / 3.0}
    if miner is fpgrowth:
        expected.update({frozenset([0]): 2.0 / 3.0, frozenset([1]): 2.0 / 3.0})
    assert _supports(actual) == expected
    assert_frame_equal(data, original)
