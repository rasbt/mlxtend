# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

from mlxtend.feature_selection import ColumnSelector


def test_ColumnSelector():
    X1_in = np.ones((4, 8))
    X1_out = ColumnSelector(cols=(1, 3)).transform(X1_in)
    assert X1_out.shape == (4, 2)


def test_ColumnSelector_drop_axis():
    X1_in = np.ones((4, 8))
    X1_out = ColumnSelector(cols=1, drop_axis=True).transform(X1_in)
    assert X1_out.shape == (4,)

    X1_out = ColumnSelector(cols=(1,), drop_axis=True).transform(X1_in)
    assert X1_out.shape == (4,)

    X1_out = ColumnSelector(cols=1).transform(X1_in)
    assert X1_out.shape == (4, 1)

    X1_out = ColumnSelector(cols=(1,)).transform(X1_in)
    assert X1_out.shape == (4, 1)


def test_ColumnSelector_in_gridsearch():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target
    pipe = make_pipeline(
        ColumnSelector(), LogisticRegression(multi_class="ovr", solver="liblinear")
    )
    grid = {
        "columnselector__cols": [[1, 2], [1, 2, 3], 0, [1]],
        "logisticregression__C": [0.1, 1.0, 10.0],
    }

    if Version(sklearn_version) < Version("0.24.1"):
        gsearch1 = GridSearchCV(
            estimator=pipe, param_grid=grid, iid=False, cv=5, n_jobs=1, refit=False
        )
    else:
        gsearch1 = GridSearchCV(
            estimator=pipe, param_grid=grid, cv=5, n_jobs=1, refit=False
        )

    gsearch1.fit(X, y)
    assert gsearch1.best_params_["columnselector__cols"] == [1, 2, 3]


def test_ColumnSelector_with_dataframe():
    iris = datasets.load_iris()
    df_in = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_out = ColumnSelector(cols=("sepal length (cm)", "sepal width (cm)")).transform(
        df_in
    )
    assert df_out.shape == (150, 2)


def test_ColumnSelector_with_dataframe_and_int_columns():
    iris = datasets.load_iris()
    df_in = pd.DataFrame(iris.data, columns=iris.feature_names)
    df_out_str = ColumnSelector(
        cols=("petal length (cm)", "petal width (cm)")
    ).transform(df_in)
    df_out_int = ColumnSelector(cols=(2, 3)).transform(df_in)

    np.testing.assert_array_equal(df_out_str[:, 0], df_out_int[:, 0])
    np.testing.assert_array_equal(df_out_str[:, 1], df_out_int[:, 1])


def test_ColumnSelector_with_dataframe_drop_axis():
    iris = datasets.load_iris()
    df_in = pd.DataFrame(iris.data, columns=iris.feature_names)
    X1_out = ColumnSelector(cols=("petal length (cm)",), drop_axis=True).transform(
        df_in
    )
    assert X1_out.shape == (150,)

    X1_out = ColumnSelector(cols=("petal length (cm)",), drop_axis=True).transform(
        df_in
    )
    assert X1_out.shape == (150,)

    X1_out = ColumnSelector(cols="petal length (cm)").transform(df_in)
    assert X1_out.shape == (150, 1)

    X1_out = ColumnSelector(cols=("petal length (cm)",)).transform(df_in)
    assert X1_out.shape == (150, 1)


def test_ColumnSelector_with_dataframe_in_gridsearch():
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    pipe = make_pipeline(ColumnSelector(), LogisticRegression())
    grid = {
        "columnselector__cols": [
            ["petal length (cm)", "petal width (cm)"],
            ["sepal length (cm)", "sepal width (cm)", "petal width (cm)"],
        ],
    }

    gsearch1 = GridSearchCV(
        estimator=pipe,
        param_grid=grid,
        cv=5,
        n_jobs=1,
        scoring="accuracy",
        refit=False,
    )

    gsearch1.fit(X, y)
    assert gsearch1.best_params_["columnselector__cols"] == [
        "petal length (cm)",
        "petal width (cm)",
    ]
