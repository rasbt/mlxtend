# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_array_equal
from scipy.sparse import csr_matrix
from mlxtend.utils import assert_raises
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from pandas import __version__ as pandas_version
from distutils.version import LooseVersion as Version
import sys
from contextlib import contextmanager
from io import StringIO


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


class FPTestEdgeCases(object):
    """
    Base class for testing edge cases for pattern mining.
    """

    def setUp(self, fpalgo):
        self.fpalgo = fpalgo

    def test_empty(self):
        df = pd.DataFrame([[]])
        res_df = self.fpalgo(df)
        expect = pd.DataFrame([], columns=['support', 'itemsets'])
        compare_dataframes(res_df, expect)


class FPTestErrors(object):
    """
    Base class for testing expected errors for pattern mining.
    """

    def setUp(self, fpalgo):
        self.one_ary = np.array(
            [[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
             [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
             [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
             [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])
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
        def test_with_dataframe(df):
            assert_raises(ValueError,
                          'The allowed values for a DataFrame are True, '
                          'False, 0, 1. Found value 2',
                          self.fpalgo, df)
        df2 = pd.DataFrame(self.one_ary, columns=self.cols).copy()
        df2.iloc[3, 3] = 2
        test_with_dataframe(df2)

        if Version(pandas_version) >= Version("1.00"):

            sdf = df2.astype(pd.SparseDtype("int", np.nan)).sparse.to_coo()
        else:
            sdf = df2.to_sparse()
        test_with_dataframe(sdf)

        if Version(pandas_version) >= Version("0.24") \
                and Version(pandas_version) <= Version("1.00"):
            sdf2 = df2.astype(pd.SparseDtype(int, fill_value=0))
            test_with_dataframe(sdf2)

    def test_sparsedataframe_notzero_column(self):

        if Version(pandas_version) < Version('1.00'):
            dfs = pd.SparseDataFrame(self.df)
        else:
            dfs = self.df.astype(pd.SparseDtype("int", np.nan)).sparse.to_coo()

        dfs.columns = [i for i in range(len(dfs.columns))]
        self.fpalgo(dfs)

        if Version(pandas_version) < Version('1.00'):
            dfs = self.df.astype(pd.SparseDtype("int", np.nan)).sparse.to_coo()
        else:
            dfs = self.df.sparse.to_coo()

        dfs.columns = [i+1 for i in range(len(dfs.columns))]
        assert_raises(ValueError,
                      'Due to current limitations in Pandas, '
                      'if the SparseDataFrame has integer column names,'
                      'names, please make sure they either start '
                      'with `0` or cast them as string column names: '
                      '`df.columns = [str(i) for i in df.columns`].',
                      self.fpalgo, dfs)


class FPTestEx1(object):
    """
    Base class for testing frequent pattern mining on a small example.
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

    def test_sparse_deprecated(self):
        def test_with_fill_values(fill_value):

            if Version(pandas_version) <= Version("1.00"):
                sdf = self.df.to_sparse(fill_value=fill_value)
                res_df = self.fpalgo(sdf, use_colnames=True)
                assert res_df.values.shape == self.fpalgo(self.df).values.shape
                assert res_df[res_df['itemsets'] == 'nothing']\
                    .values.shape == (0, 2)
                assert res_df[res_df['itemsets'] == {'Milk', 'Kidney Beans'}]\
                    .values.shape == (1, 2)
                assert res_df[res_df['itemsets'] == frozenset(
                    ('Milk', 'Kidney Beans'))]\
                    .values.shape \
                    == (1, 2)
                assert res_df[res_df['itemsets'] == frozenset(
                    ('Kidney Beans', 'Milk'))].values.shape \
                    == (1, 2)
            test_with_fill_values(0)
            test_with_fill_values(False)

    def test_sparse(self):
        if Version(pandas_version) < Version("0.24"):
            return

        def test_with_fill_values(fill_value):
            sdt = pd.SparseDtype(type(fill_value), fill_value=fill_value)
            sdf = self.df.astype(sdt)
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

    def test_sparse_with_zero(self):
        res_df = self.fpalgo(self.df)
        ary2 = self.one_ary.copy()
        ary2[3, :] = 1
        sparse_ary = csr_matrix(ary2)
        sparse_ary[3, :] = self.one_ary[3, :]

        if pandas_version < Version('1.00'):
            sdf = pd.SparseDataFrame(sparse_ary, columns=self.df.columns,
                                     default_fill_value=0)
        else:
            sdf = pd.DataFrame.sparse.from_spmatrix(sparse_ary,
                                                    columns=self.df.columns)
        res_df2 = self.fpalgo(sdf)
        compare_dataframes(res_df2, res_df)


class FPTestEx1All(FPTestEx1):
    def setUp(self, fpalgo, one_ary=None):
        FPTestEx1.setUp(self, fpalgo, one_ary=one_ary)

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

    def test_low_memory_flag(self):
        import inspect
        if 'low_memory' in inspect.signature(self.fpalgo).parameters:
            with captured_output() as (out, err):
                _ = self.fpalgo(self.df, low_memory=True, verbose=1)

            # Only get the last value of the stream to reduce test noise
            expect = 'Processing 4 combinations | Sampling itemset size 3\n'
            out = out.getvalue().split('\r')[-1]
            assert out == expect
        else:
            # If there is no low_memory argument, don't run the test.
            assert True


class FPTestEx2(object):
    """
    Base class for testing frequent pattern mining on a small example.
    """

    def setUp(self):
        database = [['a'], ['b'], ['c', 'd'], ['e']]
        te = TransactionEncoder()
        te_ary = te.fit(database).transform(database)

        self.df = pd.DataFrame(te_ary, columns=te.columns_)


class FPTestEx2All(FPTestEx2):
    def setUp(self, fpalgo):
        self.fpalgo = fpalgo
        FPTestEx2.setUp(self)

    def test_output(self):
        res_df = self.fpalgo(self.df, min_support=0.001, use_colnames=True)
        expect = pd.DataFrame([[0.25, frozenset(['a'])],
                               [0.25, frozenset(['b'])],
                               [0.25, frozenset(['c'])],
                               [0.25, frozenset(['d'])],
                               [0.25, frozenset(['e'])],
                               [0.25, frozenset(['c', 'd'])]],
                              columns=['support', 'itemsets'])

        compare_dataframes(res_df, expect)


class FPTestEx3(object):
    """
    Base class for testing frequent pattern mining on a small example.
    """

    def setUp(self):
        database = [['a'], ['b'], ['c', 'd'], ['e']]
        te = TransactionEncoder()
        te_ary = te.fit(database).transform(database)

        self.df = pd.DataFrame(te_ary, columns=te.columns_)


class FPTestEx3All(FPTestEx3):
    def setUp(self, fpalgo):
        self.fpalgo = fpalgo
        FPTestEx3.setUp(self)

    def test_output3(self):
        assert_raises(ValueError,
                      '`min_support` must be a positive '
                      'number within the interval `(0, 1]`. Got 0.0.',
                      self.fpalgo,
                      self.df,
                      min_support=0.)


def compare_dataframes(df1, df2):
    itemsets1 = [sorted(list(i)) for i in df1['itemsets']]
    itemsets2 = [sorted(list(i)) for i in df2['itemsets']]

    rows1 = sorted(zip(itemsets1, df1['support']))
    rows2 = sorted(zip(itemsets2, df2['support']))

    for r1, r2 in zip(rows1, rows2):
        assert_array_equal(r1, r2)
