import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_raises as numpy_assert_raises

from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

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
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    "jaccard",
    "certainty",
    "kulczynski",
]


# fmt: off
def test_default():
    res_df = association_rules(df_freq_items, len(df))
    res_df["antecedents"] = res_df["antecedents"].apply(lambda x: str(frozenset(x)))
    res_df["consequents"] = res_df["consequents"].apply(lambda x: str(frozenset(x)))
    res_df.sort_values(columns_ordered, inplace=True)
    res_df.reset_index(inplace=True, drop=True)

    expect = pd.DataFrame(
        [
            [(8,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 0.0, np.inf, 0, 0.6, 0.0, 0.8],
            [(6,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 0.0, np.inf, 0, 0.6, 0.0, 0.8],
            [(8, 3), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 0.0, np.inf, 0, 0.6, 0.0, 0.8],
            [(8, 5), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 1.0, 0.12, np.inf, 0.5, 0.75, 1.0, 0.875],
            [(8,), (3, 5), 0.6, 0.8, 0.6, 1.0, 1.25, 1.0, 0.12, np.inf, 0.5, 0.75, 1.0, 0.875],
            [(3,), (5,), 0.8, 1.0, 0.8, 1.0, 1.0, 1.0, 0.0, np.inf, 0, 0.8, 0.0, 0.9],
            [(5,), (3,), 1.0, 0.8, 0.8, 0.8, 1.0, 1.0, 0.0, 1.0, 0.0, 0.8, 0.0, 0.9],
            [(10,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 1.0, 0.0, np.inf, 0, 0.6, 0.0, 0.8],
            [(8,), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 1.0, 0.12, np.inf, 0.5, 0.75, 1.0, 0.875],
        ],

        columns=columns_ordered,
    )

    expect["antecedents"] = expect["antecedents"].apply(lambda x: str(frozenset(x)))
    expect["consequents"] = expect["consequents"].apply(lambda x: str(frozenset(x)))
    expect.sort_values(columns_ordered, inplace=True)
    expect.reset_index(inplace=True, drop=True)
    assert res_df.equals(expect), res_df
# fmt: on


def test_nullability():
    rows, columns = df.shape
    nan_idxs = list(range(rows)) + list(range(3, 0, -1)) + list(range(3))
    for i, j in zip(nan_idxs, range(columns)):
        df.iloc[i, j] = np.nan

    df_fp_items = fpgrowth(df, min_support=0.6, null_values=True)
    res_df = association_rules(
        df_fp_items, len(df), df, null_values=True, min_threshold=0.6
    )
    res_df["antecedents"] = res_df["antecedents"].apply(lambda x: str(frozenset(x)))
    res_df["consequents"] = res_df["consequents"].apply(lambda x: str(frozenset(x)))
    res_df.sort_values(columns_ordered, inplace=True)
    res_df.reset_index(inplace=True, drop=True)
    res_df = round(res_df, 3)

    expect = pd.DataFrame(
        [
            [
                (10, 3),
                (5,),
                0.667,
                1.0,
                0.667,
                1.0,
                1.0,
                0.6,
                0.0,
                np.inf,
                0,
                0.667,
                0,
                0.833,
            ],
            [
                (10, 5),
                (3,),
                0.667,
                1.0,
                0.667,
                1.0,
                1.0,
                0.6,
                0.0,
                np.inf,
                0,
                0.667,
                0.0,
                0.833,
            ],
            [
                (10,),
                (3, 5),
                0.75,
                1.0,
                0.667,
                1.0,
                1.0,
                0.6,
                -0.083,
                np.inf,
                -0.333,
                0.615,
                0.0,
                0.833,
            ],
            [
                (10,),
                (3,),
                0.75,
                1.0,
                0.667,
                1.0,
                1.0,
                0.6,
                -0.083,
                np.inf,
                -0.333,
                0.615,
                0.0,
                0.833,
            ],
            [
                (10,),
                (5,),
                0.75,
                1.0,
                0.667,
                1.0,
                1.0,
                0.6,
                -0.083,
                np.inf,
                -0.333,
                0.615,
                0,
                0.833,
            ],
            [
                (3, 5),
                (10,),
                1.0,
                0.75,
                0.667,
                0.667,
                0.889,
                0.6,
                -0.083,
                0.75,
                -1.0,
                0.615,
                -0.333,
                0.833,
            ],
            [
                (3,),
                (10, 5),
                1.0,
                0.667,
                0.667,
                0.667,
                1.0,
                0.6,
                0.0,
                1.0,
                0,
                0.667,
                0.0,
                0.833,
            ],
            [
                (3,),
                (10,),
                1.0,
                0.75,
                0.667,
                0.667,
                0.889,
                0.6,
                -0.083,
                0.75,
                -1.0,
                0.615,
                -0.333,
                0.833,
            ],
            [(3,), (5,), 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.0, np.inf, 0, 1.0, 0, 1.0],
            [
                (5,),
                (10, 3),
                1.0,
                0.667,
                0.667,
                0.667,
                1.0,
                0.6,
                0.0,
                1.0,
                0,
                0.667,
                0,
                0.833,
            ],
            [
                (5,),
                (10,),
                1.0,
                0.75,
                0.667,
                0.667,
                0.889,
                0.6,
                -0.083,
                0.75,
                -1.0,
                0.615,
                -0.333,
                0.833,
            ],
            [(5,), (3,), 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.0, np.inf, 0, 1.0, 0.0, 1.0],
        ],
        columns=columns_ordered,
    )

    expect["antecedents"] = expect["antecedents"].apply(lambda x: str(frozenset(x)))
    expect["consequents"] = expect["consequents"].apply(lambda x: str(frozenset(x)))
    expect.sort_values(columns_ordered, inplace=True)
    expect.reset_index(inplace=True, drop=True)
    assert res_df.equals(expect), res_df


def test_datatypes():
    res_df = association_rules(df_freq_items, len(df))
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

    res_df = association_rules(df_freq_items, len(df))
    for i in res_df["antecedents"]:
        assert isinstance(i, frozenset) is True

    for i in res_df["consequents"]:
        assert isinstance(i, frozenset) is True


def test_no_support_col():
    df_no_support_col = df_freq_items.loc[:, ["itemsets"]]
    numpy_assert_raises(ValueError, association_rules, df_no_support_col, len(df))


def test_no_itemsets_col():
    df_no_itemsets_col = df_freq_items.loc[:, ["support"]]
    numpy_assert_raises(ValueError, association_rules, df_no_itemsets_col, len(df))


def test_wrong_metric():
    numpy_assert_raises(
        ValueError, association_rules, df_freq_items, len(df), None, False, "unicorn"
    )


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
            "representativity",
            "leverage",
            "conviction",
            "zhangs_metric",
            "jaccard",
            "certainty",
            "kulczynski",
        ]
    )
    res_df = association_rules(df_freq_items, len(df), min_threshold=2)
    assert res_df.equals(expect)


def test_leverage():
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.1, metric="leverage"
    )
    assert res_df.values.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.1, metric="leverage"
    )
    assert res_df.values.shape[0] == 6


def test_conviction():
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=1.5, metric="conviction"
    )
    assert res_df.values.shape[0] == 11

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=1.5, metric="conviction"
    )
    assert res_df.values.shape[0] == 11


def test_lift():
    res_df = association_rules(df_freq_items, len(df), min_threshold=1.1, metric="lift")
    assert res_df.values.shape[0] == 6

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=1.1, metric="lift"
    )
    assert res_df.values.shape[0] == 6


def test_confidence():
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.8, metric="confidence"
    )
    assert res_df.values.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, metric="confidence"
    )
    assert res_df.values.shape[0] == 9


def test_representativity():
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=1.0, metric="representativity"
    )
    assert res_df.values.shape[0] == 16

    res_df = association_rules(
        df_freq_items_with_colnames,
        len(df),
        min_threshold=1.0,
        metric="representativity",
    )
    assert res_df.values.shape[0] == 16


def test_jaccard():
    res_df = association_rules(
        df_freq_items, len(df), min_threshold=0.7, metric="jaccard"
    )
    assert res_df.values.shape[0] == 8

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.7, metric="jaccard"
    )
    assert res_df.values.shape[0] == 8


def test_certainty():
    res_df = association_rules(
        df_freq_items, len(df), metric="certainty", min_threshold=0.6
    )
    assert res_df.values.shape[0] == 3

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), metric="certainty", min_threshold=0.6
    )
    assert res_df.values.shape[0] == 3


def test_kulczynski():
    res_df = association_rules(
        df_freq_items, len(df), metric="kulczynski", min_threshold=0.9
    )
    assert res_df.values.shape[0] == 2

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), metric="kulczynski", min_threshold=0.6
    )
    assert res_df.values.shape[0] == 16


def test_frozenset_selection():
    res_df = association_rules(df_freq_items, len(df))

    sel = res_df[res_df["consequents"] == frozenset((3, 5))]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["consequents"] == frozenset((5, 3))]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["consequents"] == {3, 5}]
    assert sel.values.shape[0] == 1

    sel = res_df[res_df["antecedents"] == frozenset((8, 3))]
    assert sel.values.shape[0] == 1


def test_override_metric_with_support():
    res_df = association_rules(df_freq_items_with_colnames, len(df), min_threshold=0.8)
    # default metric is confidence
    assert res_df.values.shape[0] == 9

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, metric="support"
    )
    assert res_df.values.shape[0] == 2

    res_df = association_rules(
        df_freq_items_with_colnames, len(df), min_threshold=0.8, support_only=True
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

    df_missing = pd.DataFrame(dict)

    numpy_assert_raises(KeyError, association_rules, df_missing, len(df))


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

    df_missing = pd.DataFrame(dict)
    df_result = association_rules(
        df_missing, len(df), support_only=True, min_threshold=0.1
    )

    assert df_result["support"].shape == (18,)
    assert int(np.isnan(df_result["support"].values).any()) != 1


def test_with_empty_dataframe():
    df_freq = df_freq_items_with_colnames.iloc[:0]
    with pytest.raises(ValueError):
        association_rules(df_freq, len(df))
