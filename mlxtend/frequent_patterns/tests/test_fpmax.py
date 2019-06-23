import unittest
import numpy as np
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestMaximal
from mlxtend.frequent_patterns import fpmax


class TestFPMax(unittest.TestCase, FPTestMaximal):
    def setUp(self):
        FPTestMaximal.setUp(self, fpmax)


class TestFPMax2(unittest.TestCase, FPTestMaximal):
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
        FPTestMaximal.setUp(self, fpmax, one_ary=one_ary)
