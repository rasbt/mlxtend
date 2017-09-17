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

columns_ordered = ['antecedants', 'consequents',
                   'antecedent support', 'consequent support',
                   'support',
                   'confidence', 'lift', 'leverage']


def test_default():
    res_df = association_rules(df_freq_items)
    res_df['antecedants'] = res_df['antecedants'].apply(
        lambda x: str(frozenset(x)))
    res_df['consequents'] = res_df['consequents'].apply(
        lambda x: str(frozenset(x)))
    res_df.sort_values(columns_ordered, inplace=True)
    res_df.reset_index(inplace=True, drop=True)

    expect = pd.DataFrame([
        [(8,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.00],
        [(6,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.00],
        [(8, 3), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.00],
        [(8, 5), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12],
        [(8,), (3, 5), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12],
        [(3,), (5,), 0.8, 1.0, 0.8, 1.0, 1.0, 0.00],
        [(5,), (3,), 1.0, 0.8, 0.8, 0.8, 1.0, 0.00],
        [(10,), (5,), 0.6, 1.0, 0.6, 1.0, 1.0, 0.00],
        [(8,), (3,), 0.6, 0.8, 0.6, 1.0, 1.25, 0.12]],
        columns=columns_ordered
    )

    expect['antecedants'] = expect['antecedants'].apply(
        lambda x: str(frozenset(x)))
    expect['consequents'] = expect['consequents'].apply(
        lambda x: str(frozenset(x)))
    expect.sort_values(columns_ordered, inplace=True)
    expect.reset_index(inplace=True, drop=True)

    print(res_df)
    assert res_df.equals(expect)


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
                 'confidence', 'lift', 'leverage']
    )
    res_df = association_rules(df_freq_items, min_threshold=2)

    assert res_df.equals(expect)
