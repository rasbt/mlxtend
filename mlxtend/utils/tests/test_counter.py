# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.utils import Counter


def test_counter():
    cnt = Counter()
    for i in range(20):
        cnt.update()
    assert cnt.curr_iter == 20
