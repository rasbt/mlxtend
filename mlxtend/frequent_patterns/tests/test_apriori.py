# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import unittest
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestAll
from mlxtend.frequent_patterns import apriori


class TestApriori(unittest.TestCase, FPTestAll):
    def setUp(self):
        FPTestAll.setUp(self, apriori)
