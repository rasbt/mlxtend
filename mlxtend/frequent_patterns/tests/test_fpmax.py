from mlxtend.frequent_patterns.tests.test_fpbase import FPTestMaximal
from mlxtend.frequent_patterns import fpmax


class TestFPMax(FPTestMaximal):
    def setUp(self):
        FPTestMaximal.setUp(self, fpmax)
