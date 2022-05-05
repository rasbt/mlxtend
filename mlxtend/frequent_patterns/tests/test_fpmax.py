import unittest

import numpy as np
import pandas as pd
from test_fpbase import (
    FPTestEdgeCases,
    FPTestErrors,
    FPTestEx1,
    FPTestEx2,
    FPTestEx3All,
    compare_dataframes,
)

from mlxtend.frequent_patterns import fpmax


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self):
        FPTestEdgeCases.setUp(self, fpmax)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self):
        FPTestErrors.setUp(self, fpmax)


class TestEx1(unittest.TestCase, FPTestEx1):
    def setUp(self, one_ary=None):
        FPTestEx1.setUp(self, fpmax, one_ary=one_ary)

    def test_default(self):
        res_df = fpmax(self.df)
        expect = pd.DataFrame(
            [
                [0.6, frozenset([5, 6])],
                [0.6, frozenset([5, 10])],
                [0.6, frozenset([3, 5, 8])],
            ],
            columns=["support", "itemsets"],
        )

        compare_dataframes(res_df, expect)

    def test_max_len(self):
        res_df1 = fpmax(self.df)
        max_len = np.max(res_df1["itemsets"].apply(len))
        assert max_len == 3

        res_df2 = fpmax(self.df, max_len=2)
        max_len = np.max(res_df2["itemsets"].apply(len))
        assert max_len == 2


class TestEx1BoolInput(TestEx1):
    def setUp(self):
        one_ary = np.array(
            [
                [False, False, False, True, False, True, True, True, True, False, True],
                [False, False, True, True, False, True, False, True, True, False, True],
                [
                    True,
                    False,
                    False,
                    True,
                    False,
                    True,
                    True,
                    False,
                    False,
                    False,
                    False,
                ],
                [
                    False,
                    True,
                    False,
                    False,
                    False,
                    True,
                    True,
                    False,
                    False,
                    True,
                    True,
                ],
                [
                    False,
                    True,
                    False,
                    True,
                    True,
                    True,
                    False,
                    False,
                    True,
                    False,
                    False,
                ],
            ]
        )
        FPTestEx1.setUp(self, fpmax, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2):
    def setUp(self):
        FPTestEx2.setUp(self)

    def test_output(self):
        res_df = fpmax(self.df, min_support=0.001, use_colnames=True)
        expect = pd.DataFrame(
            [
                [0.25, frozenset(["a"])],
                [0.25, frozenset(["b"])],
                [0.25, frozenset(["c", "d"])],
                [0.25, frozenset(["e"])],
            ],
            columns=["support", "itemsets"],
        )

        compare_dataframes(res_df, expect)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self):
        FPTestEx3All.setUp(self, fpmax)


class TestEx4(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [[1, 1, 0], [1, 0, 1], [0, 0, 1]], columns=["a", "b", "c"]
        )
        self.fpalgo = fpmax

    def test_output(self):
        res_df = self.fpalgo(self.df, min_support=0.01, use_colnames=True)
        expect = pd.DataFrame(
            [
                [0.3333333333333333, frozenset(["a", "b"])],
                [0.3333333333333333, frozenset(["a", "c"])],
            ],
            columns=["support", "itemsets"],
        )

        compare_dataframes(res_df, expect)
