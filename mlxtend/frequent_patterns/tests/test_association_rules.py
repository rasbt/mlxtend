import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from numpy.testing import assert_array_equal, assert_raises

one_ary = np.array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream', 'Kidney Beans', 'Milk',
        'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

df = pd.DataFrame(one_ary, columns=cols)

df_freq_items = apriori(df, min_support=0.6)


def test_default():
    res_df = association_rules(df_freq_items)
    expect = pd.DataFrame([
        [(8), (5), 0.6, 1.0, 1.0],
        [(6), (5), 0.6, 1.0, 1.0],
        [(8, 3), (5), 0.6, 1.0, 1.0],
        [(8, 5), (3), 0.6, 1.0, 1.25],
        [(8), (3, 5), 0.6, 1.0, 1.25],
        [(3), (5), 0.8, 1.0, 1.0],
        [(5), (3), 1.0, 0.8, 1.0],
        [(10), (5), 0.6, 1.0, 1.0],
        [(8), (3), 0.6, 1.0, 1.25]],
        columns=['antecedants', 'consequents', 'support', 'confidence', 'lift']
    )

    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)


def test_no_support_col():
    df_no_support_col = df_freq_items.loc[:, ['itemsets']]
    assert_raises(ValueError, association_rules, df_no_support_col)


def test_no_itemsets_col():
    df_no_itemsets_col = df_freq_items.loc[:, ['support']]
    assert_raises(ValueError, association_rules, df_no_itemsets_col)


def test_wrong_metric():
    assert_raises(ValueError, association_rules, df_freq_items, 'unicorn')


def test_empty_result():
    expect = pd.DataFrame(
        columns=['antecedants', 'consequents', 'support', 'confidence', 'lift']
    )
    res_df = association_rules(df_freq_items, min_threshold=2)

    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)
