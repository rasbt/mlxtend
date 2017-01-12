# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.utils import assert_raises


def test_without_message():
    def my_func():
        raise AttributeError
    assert_raises(AttributeError, func=my_func, message=None)


def test_with_message():
        def my_func():
            raise AttributeError('Failed')
        assert_raises(AttributeError, func=my_func, message='Failed')
