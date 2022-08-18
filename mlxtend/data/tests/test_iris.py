import os
import re

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.data import iris_data

this_dir, this_filename = os.path.split(__file__)
this_dir = re.sub("^stset", "", this_dir[::-1])[::-1]
DATA_PATH = os.path.join(this_dir, "data", "iris.csv.gz")


def test_iris_data_uci():
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
    original_uci_data_x, original_uci_data_y = tmp[:, :-1], tmp[:, -1]
    original_uci_data_y = original_uci_data_y.astype(int)
    iris_x, iris_y = iris_data()
    assert_array_equal(original_uci_data_x, iris_x)
    assert_array_equal(original_uci_data_y, iris_y)


def test_iris_data_r():
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
    original_r_data_x, original_r_data_y = tmp[:, :-1], tmp[:, -1]
    original_r_data_y = original_r_data_y.astype(int)
    original_r_data_x[34] = [4.9, 3.1, 1.5, 0.2]
    original_r_data_x[37] = [4.9, 3.6, 1.4, 0.1]
    iris_x, iris_y = iris_data(version="corrected")
    assert_array_equal(original_r_data_x, iris_x)


def test_iris_invalid_choice():
    with pytest.raises(ValueError) as excinfo:
        iris_data(version="bla")
        assert excinfo.value.message == "version must be 'uci' or 'corrected'."
