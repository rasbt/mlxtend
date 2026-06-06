# Sebastian Raschka 2014-2026
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

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns.apriori import generate_new_combinations


def _sorted_itemsets_with_support(df):
    rows = []
    for support, itemset in zip(df["support"], df["itemsets"]):
        rows.append((tuple(sorted(itemset)), support))
    return sorted(rows)


def _assert_itemsets_equal(result, expected):
    result_rows = _sorted_itemsets_with_support(result)
    expected_rows = _sorted_itemsets_with_support(expected)

    assert [row[0] for row in result_rows] == [row[0] for row in expected_rows]
    np.testing.assert_allclose(
        [row[1] for row in result_rows],
        [row[1] for row in expected_rows],
    )


def apriori_wrapper_low_memory(*args, **kwargs):
    return apriori(*args, **kwargs, low_memory=True)


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self):
        FPTestEdgeCases.setUp(self, apriori)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self):
        FPTestErrors.setUp(self, apriori)


class TestApriori(unittest.TestCase, FPTestEx1All):
    def setUp(self):
        FPTestEx1All.setUp(self, apriori)


class TestAprioriLowMemory(unittest.TestCase, FPTestEx1All):
    def setUp(self):
        FPTestEx1All.setUp(self, apriori_wrapper_low_memory)


class TestAprioriBoolInput(unittest.TestCase, FPTestEx1All):
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
        FPTestEx1All.setUp(self, apriori, one_ary=one_ary)


class TestAprioriSpecific(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                [1, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            columns=["A", "B", "C", "D"],
        )

    def test_n_jobs_matches_single_threaded_result(self):
        res_single = apriori(
            self.df, min_support=0.4, use_colnames=True, max_len=3, n_jobs=1
        )
        res_parallel = apriori(
            self.df, min_support=0.4, use_colnames=True, max_len=3, n_jobs=2
        )

        _assert_itemsets_equal(res_parallel, res_single)

    def test_low_memory_matches_default_result(self):
        res_default = apriori(self.df, min_support=0.4, use_colnames=True, max_len=3)
        res_low_memory = apriori(
            self.df, min_support=0.4, use_colnames=True, max_len=3, low_memory=True
        )

        _assert_itemsets_equal(res_low_memory, res_default)

    def test_max_len_one_returns_only_single_itemsets(self):
        result = apriori(self.df, min_support=0.6, use_colnames=True, max_len=1)
        expected = pd.DataFrame(
            [
                [0.8, frozenset(["A"])],
                [0.8, frozenset(["B"])],
                [0.8, frozenset(["C"])],
                [0.8, frozenset(["D"])],
            ],
            columns=["support", "itemsets"],
        )

        _assert_itemsets_equal(result, expected)

    def test_min_support_one_includes_itemsets_present_in_every_transaction(self):
        df = pd.DataFrame(
            [
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 0],
            ],
            columns=["A", "B", "C"],
        )

        result = apriori(df, min_support=1.0, use_colnames=True)
        expected = pd.DataFrame(
            [
                [1.0, frozenset(["A"])],
                [1.0, frozenset(["B"])],
                [1.0, frozenset(["A", "B"])],
            ],
            columns=["support", "itemsets"],
        )

        _assert_itemsets_equal(result, expected)

    def test_returns_empty_result_if_no_item_meets_min_support(self):
        df = pd.DataFrame(
            [
                [1, 0],
                [0, 1],
            ],
            columns=["A", "B"],
        )

        result = apriori(df, min_support=1.0, use_colnames=True)

        assert result.columns.tolist() == ["support", "itemsets"]
        assert result.empty

    def test_candidate_generation_joins_only_matching_prefixes(self):
        old_combinations = np.array(
            [
                [0, 1],
                [0, 2],
                [1, 2],
                [1, 3],
                [2, 3],
            ]
        )

        candidates = np.fromiter(
            generate_new_combinations(old_combinations), dtype=int
        ).reshape(-1, 3)

        np.testing.assert_array_equal(candidates, np.array([[0, 1, 2], [1, 2, 3]]))

    def test_candidate_generation_prunes_missing_subsets(self):
        old_combinations = np.array(
            [
                [0, 1, 2],
                [0, 1, 3],
                [0, 2, 3],
            ]
        )

        candidates = np.fromiter(generate_new_combinations(old_combinations), dtype=int)

        assert candidates.size == 0


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self):
        FPTestEx2All.setUp(self, apriori)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self):
        FPTestEx3All.setUp(self, apriori)
