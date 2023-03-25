# Sebastian Raschka 2014-2022
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import unittest

import numpy as np
import pandas as pd
from test_fpbase import (
    FPTestEdgeCases,
    FPTestErrors,
    FPTestEx1All,
    FPTestEx2All,
    FPTestEx3All,
)

from mlxtend.frequent_patterns import apriori, fpgrowth, hmine


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self):
        FPTestEdgeCases.setUp(self, hmine)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self):
        FPTestErrors.setUp(self, hmine)


class TestHmine(unittest.TestCase, FPTestEx1All):
    def setUp(self):
        FPTestEx1All.setUp(self, hmine)


class TestHmineBoolInput(unittest.TestCase, FPTestEx1All):
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
        FPTestEx1All.setUp(self, hmine, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self):
        FPTestEx2All.setUp(self, hmine)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self):
        FPTestEx3All.setUp(self, hmine)


class TestCorrect(unittest.TestCase):
    def setUp(self):
        self.one_ary = np.array([
                                [False, False, False, True, False, True, True, True, True, False, True],
                                [False, False, True, True, False, True, False, True, True, False, True],
                                [True, False, False, True, False, True, True, False, False, False, False],
                                [False, True, False, False, False, True, True, False, False, True, True],
                                [False, True, False, True, True, True, False, False, True, False, False],
                                ])
        self.cols = [
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
        self.df = pd.DataFrame(self.one_ary, columns=self.cols)

    def test_compare_correct(self):

        expect = pd.DataFrame(
            [
                [0.8, np.array([3])],
                [1.0, np.array([5])],
                [0.6, np.array([6])],
                [0.6, np.array([8])],
                [0.6, np.array([10])],
                [0.8, np.array([3, 5])],
                [0.6, np.array([3, 8])],
                [0.6, np.array([5, 6])],
                [0.6, np.array([5, 8])],
                [0.6, np.array([5, 10])],
                [0.6, np.array([3, 5, 8])],
            ],
            columns=["support", "itemsets"],
        )
        algorithms = {hmine: None, fpgrowth: None, apriori: None}
        for algo in algorithms.keys():
            self.setUp()
            res_df = algo(self.df, min_support=0.6)
            compare_df(res_df, expect)
            algorithms[algo] = res_df

        compare_df(algorithms[hmine], algorithms[fpgrowth])
        compare_df(algorithms[hmine], algorithms[apriori])
        compare_df(algorithms[fpgrowth], algorithms[apriori])


def compare_df(df1, df2):
    itemsets1 = [sorted(list(i)) for i in df1["itemsets"]]
    itemsets2 = [sorted(list(i)) for i in df2["itemsets"]]
    rows1 = sorted(zip(itemsets1, df1["support"]))
    rows2 = sorted(zip(itemsets2, df2["support"]))
    for row1, row2 in zip(rows1, rows2):
        if row1[0] != row2[0]:
            msg = f"Expected different frequent itemsets\nx:{row1[0]}\ny:{row2[0]}"
            raise AssertionError(msg)
        elif row1[1] != row2[1]:
            msg = f"Expected different support\nx:{row1[1]}\ny:{row2[1]}"
            raise AssertionError(msg)
