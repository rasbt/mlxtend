# mlxtend Machine Learning Library Extensions
# Author: Fatih Sen <fatih.sn2000@gmail.com>
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
    compare_dataframes,
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
        self.one_ary = np.array(
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
            compare_dataframes(res_df, expect)
            algorithms[algo] = res_df

        compare_dataframes(algorithms[hmine], algorithms[fpgrowth])
        compare_dataframes(algorithms[hmine], algorithms[apriori])
        compare_dataframes(algorithms[fpgrowth], algorithms[apriori])
