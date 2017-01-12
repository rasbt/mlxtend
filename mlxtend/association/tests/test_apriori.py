# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.association import apriori
from numpy.testing import assert_array_equal
import pandas as pd

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

one_ary = np.array([[0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1],
                    [0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1],
                    [0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]])

cols = ['Apple', 'Corn', 'Dill', 'Eggs', 'Ice cream', 'Kidney Beans', 'Milk',
        'Nutmeg', 'Onion', 'Unicorn', 'Yogurt']

df = pd.DataFrame(one_ary, columns=cols)


def test_default():
    res_df = apriori(df, min_support=0.6)
    expect = pd.DataFrame([[0.8, np.array([3]), 1],
                           [1.0, np.array([5]), 1],
                           [0.6, np.array([6]), 1],
                           [0.6, np.array([8]), 1],
                           [0.6, np.array([10]), 1],
                           [0.8, np.array([3, 5]), 2],
                           [0.6, np.array([3, 8]), 2],
                           [0.6, np.array([5, 6]), 2],
                           [0.6, np.array([5, 8]), 2],
                           [0.6, np.array([5, 10]), 2],
                           [0.6, np.array([3, 5, 8]), 3]],
                          columns=['support', 'itemsets', 'length'])

    for a, b in zip(res_df, expect):
        assert_array_equal(a, b)
