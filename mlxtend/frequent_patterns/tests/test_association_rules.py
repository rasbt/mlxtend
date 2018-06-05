import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from numpy.testing import assert_raises

one_ary = np.array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream', 'Kidney Beans', 'Milk',
        'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

df = pd.DataFrame(one_ary, columns=cols)

df_freq_items = apriori(df, min_support=0.6)

df_freq_items_with_colnames = apriori(df, min_support=0.6, use_colnames=True)

columns_ordered = ['antecedants', 'consequents',
                   'antecedent support', 'consequent support',
                   'support',
                   'confidence', 'lift', 'leverage', 'conviction']


def test_default():
    res_df = association_rules(df_freq_items)
    res_df['antecedants'] = res_df['antecedants'].apply(
        lambda x: str(frozenset(x)))
    res_df['consequents'] = res_df['consequents'].apply(
        lambda x: str(frozenset(x)))
    res_df.sort_values(columns_ordered, inplace=True)
    res_df.reset_index(inplace=True, drop=True)

    expect = pd.DataFrame([
        [(8,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf],
        [(6,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf],
        [(8, 3), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf],
        [(8, 5), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf],
        [(8,), (3, 5), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf],
        [(3,), (5,), 0.8, 1.0, 0.8, 1.0, 1.0, 0.0, np.inf],
        [(5,), (3,), 1.0, 0.8, 0.8, 0.8, 1.0, 0.0, 1.0],
        [(10,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.0, np.inf],
        [(8,), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12, np.inf]],
        columns=columns_ordered
    )

    expect['antecedants'] = expect['antecedants'].apply(
        lambda x: str(frozenset(x)))
    expect['consequents'] = expect['consequents'].apply(
        lambda x: str(frozenset(x)))
    expect.sort_values(columns_ordered, inplace=True)
    expect.reset_index(inplace=True, drop=True)

    assert res_df.equals(expect), res_df


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
        columns=['antecedants', 'consequents',
                 'antecedent support',
                 'consequent support',
                 'support',
                 'confidence', 'lift', 'leverage',
                 'conviction']
    )
    res_df = association_rules(df_freq_items, min_threshold=2)

    assert res_df.equals(expect)


def test_leverage():
    res_df = association_rules(df_freq_items,
                               min_threshold=0.1,
                               metric='leverage')
    assert res_df.values.shape[0] == 6

    res_df = association_rules(df_freq_items_with_colnames,
                               min_threshold=0.1,
                               metric='leverage')
    assert res_df.values.shape[0] == 6


def test_conviction():
    res_df = association_rules(df_freq_items,
                               min_threshold=1.5,
                               metric='conviction')
    assert res_df.values.shape[0] == 11

    res_df = association_rules(df_freq_items_with_colnames,
                               min_threshold=1.5,
                               metric='conviction')
    assert res_df.values.shape[0] == 11


def test_lift():
    res_df = association_rules(df_freq_items,
                               min_threshold=1.1,
                               metric='lift')
    assert res_df.values.shape[0] == 6

    res_df = association_rules(df_freq_items_with_colnames,
                               min_threshold=1.1,
                               metric='lift')
    assert res_df.values.shape[0] == 6


def test_confidence():
    res_df = association_rules(df_freq_items,
                               min_threshold=0.8,
                               metric='confidence')
    assert res_df.values.shape[0] == 9

    res_df = association_rules(df_freq_items_with_colnames,
                               min_threshold=0.8,
                               metric='confidence')
    assert res_df.values.shape[0] == 9
