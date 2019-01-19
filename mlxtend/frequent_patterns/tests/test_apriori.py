# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np

from mlxtend.frequent_patterns import apriori
from numpy.testing import assert_array_equal
from mlxtend.utils import assert_raises
import pandas as pd

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

one_ary = np.array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream', 'Kidney Beans', 'Milk',
        'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

df = pd.DataFrame(one_ary, columns=cols)


def test_default():
    res_df = apriori(df)
    expect = pd.DataFrame([[0.8, np.array([3]), 1],
                           [1.0, np.array([5]), 1],
                           [0.6, np.array([6]), 1],
                           [0.6, np.array([8]), 1],
                           [0.6, np.array([10]), 1],
                           [0.8, np.array([3, 5]), 2],
                           [0.6, np.array([3, 8]), 2],
                           [0.6, np.array([5, 6]), 2],
                           [0.6, np.array([5, 8]), 2],
                           [0.6, np.array([5, 10]), 2],
                           [0.6, np.array([3, 5, 8]), 3]],
                          columns=['support', 'itemsets', 'length'])

    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)


def test_max_len():
    res_df1 = apriori(df)
    assert len(res_df1.iloc[-1, -1]) == 3

    res_df2 = apriori(df, max_len=2)
    assert len(res_df2.iloc[-1, -1]) == 2


def test_itemsets_type():
    res_colindice = apriori(df, use_colnames=False)  # This is default behavior
    for i in res_colindice['itemsets']:
        assert isinstance(i, frozenset) is True

    res_colnames = apriori(df, use_colnames=True)
    for i in res_colnames['itemsets']:
        assert isinstance(i, frozenset) is True


def test_frozenset_selection():
    res_df = apriori(df, use_colnames=True)
    assert res_df.values.shape == (11, 2)
    assert res_df[res_df['itemsets']
                  == 'nothing'].values.shape == (0, 2)
    assert res_df[res_df['itemsets']
                  == {'Eggs', 'Kidney Beans'}].values.shape == (1, 2)
    assert res_df[res_df['itemsets']
                  == frozenset(('Eggs', 'Kidney Beans'))].values.shape \
        == (1, 2)
    assert res_df[res_df['itemsets']
                  == frozenset(('Kidney Beans', 'Eggs'))].values.shape \
        == (1, 2)


def test_sparse_apriori():
    def test_with_fill_values(fill_value):
        sdf = df.to_sparse(fill_value=fill_value)
        res_df = apriori(sdf, use_colnames=True)
        assert res_df.values.shape == (11, 2)
        assert res_df[res_df['itemsets']
                      == 'nothing'].values.shape == (0, 2)
        assert res_df[res_df['itemsets']
                      == {'Eggs', 'Kidney Beans'}].values.shape == (1, 2)
        assert res_df[res_df['itemsets']
                      == frozenset(('Eggs', 'Kidney Beans'))].values.shape \
            == (1, 2)
        assert res_df[res_df['itemsets']
                      == frozenset(('Kidney Beans', 'Eggs'))].values.shape \
            == (1, 2)
    test_with_fill_values(0)
    test_with_fill_values(False)


def test_raise_error_if_input_is_not_binary():
    df2 = pd.DataFrame(one_ary, columns=cols).copy()
    df2.iloc[3, 3] = 2

    assert_raises(ValueError,
                  'The allowed values for a DataFrame are True, '
                  'False, 0, 1. Found value 2',
                  apriori, df2)
