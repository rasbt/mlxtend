import unittest
import numpy as np
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestAll
from mlxtend.frequent_patterns import fpgrowth


class TestFPGrowth(unittest.TestCase, FPTestAll):
    def setUp(self):
        FPTestAll.setUp(self, fpgrowth)


class TestFPGrowth2(unittest.TestCase, FPTestAll):
    def setUp(self):
        one_ary = np.array(
            [[False, False, False, True, False, True, True, True, True,
              False, True],
             [False, False, True, True, False, True, False, True, True,
              False, True],
             [True, False, False, True, False, True, True, False, False,
              False, False],
             [False, True, False, False, False, True, True, False, False,
              True, True],
             [False, True, False, True, True, True, False, False, True,
              False, False]])
        FPTestAll.setUp(self, fpgrowth, one_ary=one_ary)
