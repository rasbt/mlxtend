import unittest
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestAll
from mlxtend.frequent_patterns import fpgrowth


class TestFPGrowth(unittest.TestCase, FPTestAll):
    def setUp(self):
        FPTestAll.setUp(self, fpgrowth)
