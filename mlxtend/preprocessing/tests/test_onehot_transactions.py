# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.preprocessing import OnehotTransactions
from sklearn.base import clone
from mlxtend.utils import assert_raises


dataset = [['Apple', 'Beer', 'Rice', 'Chicken'],
           ['Apple', 'Beer', 'Rice'],
           ['Apple', 'Beer'],
           ['Apple', 'Bananas'],
           ['Milk', 'Beer', 'Rice', 'Chicken'],
           ['Milk', 'Beer', 'Rice'],
           ['Milk', 'Beer'],
           ['Apple', 'Bananas']]

data_sorted = [['Apple', 'Beer', 'Chicken', 'Rice'],
               ['Apple', 'Beer', 'Rice'],
               ['Apple', 'Beer'],
               ['Apple', 'Bananas'],
               ['Beer', 'Chicken', 'Milk', 'Rice'],
               ['Beer', 'Milk', 'Rice'],
               ['Beer', 'Milk'],
               ['Apple', 'Bananas']]


expect = np.array([[1, 0, 1, 1, 0, 1],
                   [1, 0, 1, 0, 0, 1],
                   [1, 0, 1, 0, 0, 0],
                   [1, 1, 0, 0, 0, 0],
                   [0, 0, 1, 1, 1, 1],
                   [0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 0, 1, 0],
                   [1, 1, 0, 0, 0, 0]])


def test_fit():
    oht = OnehotTransactions()
    oht.fit(dataset)
    assert(oht.columns_ == ['Apple', 'Bananas', 'Beer',
                            'Chicken', 'Milk', 'Rice'])


def test_transform():

    oht = OnehotTransactions()
    oht.fit(dataset)
    trans = oht.transform(dataset)
    np.testing.assert_array_equal(expect, trans)


def test_fit_transform():
    oht = OnehotTransactions()
    trans = oht.fit_transform(dataset)
    np.testing.assert_array_equal(expect, trans)


def test_inverse_transform():
    oht = OnehotTransactions()
    oht.fit(dataset)
    np.testing.assert_array_equal(np.array(data_sorted),
                                  np.array(oht.inverse_transform(expect)))


def test_cloning():

    oht = OnehotTransactions()
    oht.fit(dataset)
    oht2 = clone(oht)

    msg = ("'OnehotTransactions' object has no attribute 'columns_'")
    assert_raises(AttributeError,
                  msg,
                  oht2.transform,
                  dataset)

    trans = oht2.fit_transform(dataset)
    np.testing.assert_array_equal(expect, trans)
