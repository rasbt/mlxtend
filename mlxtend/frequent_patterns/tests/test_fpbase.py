# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_array_equal
from mlxtend.utils import assert_raises
import pandas as pd


class FPTestBase(object):
    """
    Base testing class for frequent pattern mining. This class should include
    setup and tests common to all methods (e.g., error for improper input)
    """

    def setUp(self, fpalgo, one_ary=None):
        if one_ary is None:
            self.one_ary = np.array(
               [[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

        else:
            self.one_ary = one_ary

        self.cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream',
                     'Kidney Beans', 'Milk',
                     'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

        self.df = pd.DataFrame(self.one_ary, columns=self.cols)

        self.fpalgo = fpalgo

    def test_itemsets_type(self):
        # This is default behavior
        res_colindice = self.fpalgo(self.df, use_colnames=False)
        for i in res_colindice['itemsets']:
            assert isinstance(i, frozenset) is True

        res_colnames = self.fpalgo(self.df, use_colnames=True)
        for i in res_colnames['itemsets']:
            assert isinstance(i, frozenset) is True

    def test_raise_error_if_input_is_not_binary(self):
        df2 = pd.DataFrame(self.one_ary, columns=self.cols).copy()
        df2.iloc[3, 3] = 2

        assert_raises(ValueError,
                      'The allowed values for a DataFrame are True, '
                      'False, 0, 1. Found value 2',
                      self.fpalgo, df2)

    def test_sparsedataframe_notzero_column(self):
        dfs = pd.SparseDataFrame(self.df)
        dfs.columns = [i for i in range(len(dfs.columns))]
        self.fpalgo(dfs)

        dfs = pd.SparseDataFrame(self.df)
        dfs.columns = [i+1 for i in range(len(dfs.columns))]
        assert_raises(ValueError,
                      'Due to current limitations in Pandas, '
                      'if the SparseDataFrame has integer column names,'
                      'names, please make sure they either start '
                      'with `0` or cast them as string column names: '
                      '`df.columns = [str(i) for i in df.columns`].',
                      self.fpalgo, dfs)

    def test_frozenset_selection(self):
        res_df = self.fpalgo(self.df, use_colnames=True)
        assert res_df.values.shape == self.fpalgo(self.df).values.shape
        assert res_df[res_df['itemsets']
                      == 'nothing'].values.shape == (0, 2)
        assert res_df[res_df['itemsets']
                      == {'Milk', 'Kidney Beans'}].values.shape == (1, 2)
        assert res_df[res_df['itemsets']
                      == frozenset(('Milk', 'Kidney Beans'))].values.shape \
            == (1, 2)
        assert res_df[res_df['itemsets']
                      == frozenset(('Kidney Beans', 'Milk'))].values.shape \
            == (1, 2)

    def test_sparse(self):
        def test_with_fill_values(fill_value):
            sdf = self.df.to_sparse(fill_value=fill_value)
            res_df = self.fpalgo(sdf, use_colnames=True)
            assert res_df.values.shape == self.fpalgo(self.df).values.shape
            assert res_df[res_df['itemsets']
                          == 'nothing'].values.shape == (0, 2)
            assert res_df[res_df['itemsets']
                          == {'Milk', 'Kidney Beans'}].values.shape == (1, 2)
            assert res_df[res_df['itemsets'] ==
                          frozenset(('Milk', 'Kidney Beans'))].values.shape \
                == (1, 2)
            assert res_df[res_df['itemsets'] ==
                          frozenset(('Kidney Beans', 'Milk'))].values.shape \
                == (1, 2)
        test_with_fill_values(0)
        test_with_fill_values(False)


class FPTestAll(FPTestBase):
    def setUp(self, fpalgo, one_ary=None):
        FPTestBase.setUp(self, fpalgo, one_ary=one_ary)

    def test_default(self):
        res_df = self.fpalgo(self.df)
        expect = pd.DataFrame([[0.8, np.array([3])],
                               [1.0, np.array([5])],
                               [0.6, np.array([6])],
                               [0.6, np.array([8])],
                               [0.6, np.array([10])],
                               [0.8, np.array([3, 5])],
                               [0.6, np.array([3, 8])],
                               [0.6, np.array([5, 6])],
                               [0.6, np.array([5, 8])],
                               [0.6, np.array([5, 10])],
                               [0.6, np.array([3, 5, 8])]],
                              columns=['support', 'itemsets'])

        compare_dataframes(res_df, expect)

    def test_max_len(self):
        res_df1 = self.fpalgo(self.df)
        max_len = np.max(res_df1['itemsets'].apply(len))
        assert max_len == 3

        res_df2 = self.fpalgo(self.df, max_len=2)
        max_len = np.max(res_df2['itemsets'].apply(len))
        assert max_len == 2


class FPTestMaximal(FPTestBase):
    def setUp(self, fpalgo, one_ary=None):
        FPTestBase.setUp(self, fpalgo, one_ary=one_ary)

    def test_default(self):
        res_df = self.fpalgo(self.df)
        expect = pd.DataFrame([[0.6, frozenset([5, 6])],
                               [0.6, frozenset([5, 10])],
                               [0.6, frozenset([3, 5, 8])]],
                              columns=['support', 'itemsets'])

        compare_dataframes(res_df, expect)

    def test_max_len(self):
        res_df1 = self.fpalgo(self.df)
        max_len = np.max(res_df1['itemsets'].apply(len))
        assert max_len == 3

        res_df2 = self.fpalgo(self.df, max_len=2)
        max_len = np.max(res_df2['itemsets'].apply(len))
        assert max_len == 2


def compare_dataframes(df1, df2):
    itemsets1 = [sorted(list(i)) for i in df1['itemsets']]
    itemsets2 = [sorted(list(i)) for i in df2['itemsets']]

    rows1 = sorted(zip(itemsets1, df1['support']))
    rows2 = sorted(zip(itemsets2, df2['support']))

    for r1, r2 in zip(rows1, rows2):
        assert_array_equal(r1, r2)
