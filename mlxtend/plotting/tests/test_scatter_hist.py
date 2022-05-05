import pandas as pd

from mlxtend.data import iris_data
from mlxtend.plotting import scatter_hist

X, y = iris_data()
df = pd.DataFrame(X)
df.columns = [
    "sepal length [cm]",
    "sepal width [cm]",
    "petal length [cm]",
    "petal width [cm]",
]


def test_pass_data_as_dataframe():
    scatter_hist(df["sepal length [cm]"], df["sepal width [cm]"])


def test_pass_data_as_numpy_array():
    scatter_hist(X[:, 0], X[:, 1])
