import unittest
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestBase
from mlxtend.frequent_patterns import fpgrowth


class FPTestGrowth(unittest.TestCase, FPTestBase):
    def setUp(self):
        FPTestBase.setUp(self)
        self.fpalgo = fpgrowth
