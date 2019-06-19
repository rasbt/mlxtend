import unittest
from mlxtend.frequent_patterns.tests.test_fpbase import FPTestMaximal
from mlxtend.frequent_patterns import fpmax


class TestFPMax(unittest.TestCase, FPTestMaximal):
    def setUp(self):
        FPTestMaximal.setUp(self, fpmax)
