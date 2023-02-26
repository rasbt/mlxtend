import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_raises as numpy_assert_raises

from mlxtend.frequent_patterns import apriori, association_rules

one_ary = np.array(
    [
        [0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    ]
)

cols = [
    "Apple",
    "Corn",
    "Dill",
    "Eggs",
    "Ice cream",
    "Kidney Beans",
    "Milk",
    "Nutmeg",
    "Onion",
    "Unicorn",
    "Yogurt",
]

df = pd.DataFrame(one_ary, columns=cols)

df_freq_items = apriori(df, min_support=0.6)

df_freq_items_with_colnames = apriori(df, min_support=0.6, use_colnames=True)

columns_ordered = [
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
    "zhangs_metric",
]


def test_default():
    res_df = association_rules(df_freq_items)
    res_df["antecedents"] = res_df["antecedents"].apply(lambda x: str(frozenset(x)))
    res_df["consequents"] = res_df["consequents"].apply(lambda x: str(frozenset(x)))
    res_df.sort_values(columns_ordered, inplace=True)
    res_df.reset_index(inplace=True, drop=True)

    expect = pd.DataFrame(
        [
            [(8,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf, 0],
            [(6,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf, 0],
            [(8, 3), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf, 0],
            [(8, 5), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf, 0.5],
            [(8,), (3, 5), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf, 0.5],
            [(3,), (5,), 0.8, 1.0, 0.8, 1.0, 1.0, 0.0, np.inf, 0],
            [(5,), (3,), 1.0, 0.8, 0.8, 0.8, 1.0, 0.0, 1.0, 0],
            [(10,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf, 0],
            [(8,), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf, 0.5],
        ],
        columns=columns_ordered,
    )

    expect["antecedents"] = expect["antecedents"].apply(lambda x: str(frozenset(x)))
    expect["consequents"] = expect["consequents"].apply(lambda x: str(frozenset(x)))
    expect.sort_values(columns_ordered, inplace=True)
    expect.reset_index(inplace=True, drop=True)

    assert res_df.equals(expect), res_df


def test_datatypes():
    res_df = association_rules(df_freq_items)
    for i in res_df["antecedents"]:
        assert isinstance(i, frozenset) is True

    for i in res_df["consequents"]:
        assert isinstance(i, frozenset) is True

    # cast itemset-containing dataframe to set and
    # check if association_rule converts it internally
    # back to frozensets
    df_freq_items_copy = df_freq_items.copy()
    df_freq_items_copy["itemsets"] = df_freq_items_copy["itemsets"].apply(
        lambda x: set(x)
    )

    res_df = association_rules(df_freq_items)
    for i in res_df["antecedents"]:
        assert isinstance(i, frozenset) is True

    for i in res_df["consequents"]:
        assert isinstance(i, frozenset) is True


def test_no_support_col():
    df_no_support_col = df_freq_items.loc[:, ["itemsets"]]
    numpy_assert_raises(ValueError, association_rules, df_no_support_col)


def test_no_itemsets_col():
    df_no_itemsets_col = df_freq_items.loc[:, ["support"]]
    numpy_assert_raises(ValueError, association_rules, df_no_itemsets_col)


def test_wrong_metric():
    numpy_assert_raises(ValueError, association_rules, df_freq_items, "unicorn")


def test_empty_result():
    expect = pd.DataFrame(
        columns=[
            "antecedents",
            "consequents",
            "antecedent support",
            "consequent support",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "zhangs_metric",
        ]
    )
    res_df = association_rules(df_freq_items, min_threshold=2)
    assert res_df.equals(expect)


def test_leverage():
    res_df = association_rules(df_freq_items, min_threshold=0.1, metric="leverage")
    assert res_df.values.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=0.1, metric="leverage"
    )
    assert res_df.values.shape[0] == 6


def test_conviction():
    res_df = association_rules(df_freq_items, min_threshold=1.5, metric="conviction")
    assert res_df.values.shape[0] == 11

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=1.5, metric="conviction"
    )
    assert res_df.values.shape[0] == 11


def test_lift():
    res_df = association_rules(df_freq_items, min_threshold=1.1, metric="lift")
    assert res_df.values.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=1.1, metric="lift"
    )
    assert res_df.values.shape[0] == 6


def test_confidence():
    res_df = association_rules(df_freq_items, min_threshold=0.8, metric="confidence")
    assert res_df.values.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=0.8, metric="confidence"
    )
    assert res_df.values.shape[0] == 9


def test_frozenset_selection():
    res_df = association_rules(df_freq_items)

    sel = res_df[res_df["consequents"] == frozenset((3, 5))]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["consequents"] == frozenset((5, 3))]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["consequents"] == {3, 5}]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["antecedents"] == frozenset((8, 3))]
    assert sel.values.shape[0] == 1


def test_override_metric_with_support():
    res_df = association_rules(df_freq_items_with_colnames, min_threshold=0.8)
    # default metric is confidence
    assert res_df.values.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=0.8, metric="support"
    )
    assert res_df.values.shape[0] == 2

    res_df = association_rules(
        df_freq_items_with_colnames, min_threshold=0.8, support_only=True
    )
    assert res_df.values.shape[0] == 2


def test_on_df_with_missing_entries():
    # this is a data frame where information about
    # antecedents and consequents have been cropped
    # see https://github.com/rasbt/mlxtend/issues/390
    # for more details
    dict = {
        "itemsets": [
            ["177", "176"],
            ["177", "179"],
            ["176", "178"],
            ["176", "179"],
            ["93", "100"],
            ["177", "178"],
            ["177", "176", "178"],
        ],
        "support": [
            0.253623,
            0.253623,
            0.217391,
            0.217391,
            0.181159,
            0.108696,
            0.108696,
        ],
    }

    df = pd.DataFrame(dict)

    numpy_assert_raises(KeyError, association_rules, df)


def test_on_df_with_missing_entries_support_only():
    # this is a data frame where information about
    # antecedents and consequents have been cropped
    # see https://github.com/rasbt/mlxtend/issues/390
    # for more details
    dict = {
        "itemsets": [
            ["177", "176"],
            ["177", "179"],
            ["176", "178"],
            ["176", "179"],
            ["93", "100"],
            ["177", "178"],
            ["177", "176", "178"],
        ],
        "support": [
            0.253623,
            0.253623,
            0.217391,
            0.217391,
            0.181159,
            0.108696,
            0.108696,
        ],
    }

    df = pd.DataFrame(dict)
    df_result = association_rules(df, support_only=True, min_threshold=0.1)

    assert df_result["support"].shape == (18,)
    assert int(np.isnan(df_result["support"].values).any()) != 1


def test_with_empty_dataframe():
    df = df_freq_items_with_colnames.iloc[:0]
    with pytest.raises(ValueError):
        association_rules(df)
