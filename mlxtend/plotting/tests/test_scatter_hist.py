from mlxtend.data import iris_data
from mlxtend.plotting import scatter_hist
import pandas as pd
import pytest

X, y = iris_data()
df = pd.DataFrame(X)
df.columns = ['sepal length [cm]', 'sepal width [cm]', 'petal length [cm]', 'petal width [cm]']


def test_pass_data_as_dataframe():
    scatter_hist("sepal length [cm]", "sepal width [cm]", df)


def test_pass_data_as_numpy_array():
    scatter_hist(0, 1, X)


def test_incorrect_x_or_y_data_as_dataframe():
    with pytest.raises(AssertionError) as execinfo:
        scatter_hist(0, "sepal width [cm]", df)
        assert execinfo.value.message == 'Assertion failed'


def test_incorrect_x_or_y_data_as_numpy_array():
    with pytest.raises(AssertionError) as execinfo:
        scatter_hist("sepal length [cm]", 1, X)
        assert execinfo.value.message == 'Assertion failed'
