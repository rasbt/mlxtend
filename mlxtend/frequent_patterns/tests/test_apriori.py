# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import unittest
import numpy as np
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestAll
from mlxtend.frequent_patterns import apriori


class TestApriori(unittest.TestCase, FPTestAll):
    def setUp(self):
        FPTestAll.setUp(self, apriori)


class TestApriori2(unittest.TestCase, FPTestAll):
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
        FPTestAll.setUp(self, apriori, one_ary=one_ary)
