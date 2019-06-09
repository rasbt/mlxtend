# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import unittest
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestBase
from mlxtend.frequent_patterns import apriori


class FPTestGrowth(unittest.TestCase, FPTestBase):
    def setUp(self):
        FPTestBase.setUp(self)
        self.fpalgo = apriori
