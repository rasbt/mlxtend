import time
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

from mlxtend.frequent_patterns import fpgrowth


class TestEdgeCases(unittest.TestCase, FPTestEdgeCases):
    def setUp(self):
        FPTestEdgeCases.setUp(self, fpgrowth)


class TestErrors(unittest.TestCase, FPTestErrors):
    def setUp(self):
        FPTestErrors.setUp(self, fpgrowth)


class TestEx1(unittest.TestCase, FPTestEx1All):
    def setUp(self):
        FPTestEx1All.setUp(self, fpgrowth)


class TestEx1BoolInput(unittest.TestCase, FPTestEx1All):
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
        FPTestEx1All.setUp(self, fpgrowth, one_ary=one_ary)


class TestEx2(unittest.TestCase, FPTestEx2All):
    def setUp(self):
        FPTestEx2All.setUp(self, fpgrowth)


class TestEx3(unittest.TestCase, FPTestEx3All):
    def setUp(self):
        FPTestEx3All.setUp(self, fpgrowth)


# Adopted from https://github.com/rasbt/mlxtend/issues/1130
def _create_dataframe(n_rows=10000, n_cols=400):
    rng = np.random.default_rng(42)

    # Generate realistic sparse boolean data with varying support levels
    support_values = np.zeros(n_cols)
    n_very_low = int(n_cols * 0.9)
    support_values[:n_very_low] = rng.uniform(0.0001, 0.009, n_very_low)

    n_medium = int(n_cols * 0.06)
    support_values[n_very_low : n_very_low + n_medium] = rng.uniform(0.01, 0.1, n_medium)

    n_high = n_cols - n_very_low - n_medium
    support_values[n_very_low + n_medium :] = rng.uniform(0.1, 0.65, n_high)

    data = {}
    for i in range(n_cols):
        col_name = f"feature_{i:04d}"
        prob = support_values[i]
        data[col_name] = rng.random(n_rows) < prob

    return pd.DataFrame(data)


def test_fpgrowth_completes_within_5_seconds():
    """Ensure fpgrowth runs within 5 seconds on a sparse dataset."""
    df = _create_dataframe()
    start_time = time.perf_counter()
    frequent_itemsets = fpgrowth(df, min_support=0.01, use_colnames=True)
    elapsed_time = time.perf_counter() - start_time

    assert (
        elapsed_time < 5
    ), (
        f"fpgrowth took {elapsed_time:.2f}s on df shape {df.shape} "
        f"and density {df.values.mean():.4f} with "
        f"{len(frequent_itemsets)} itemsets found"
    )
