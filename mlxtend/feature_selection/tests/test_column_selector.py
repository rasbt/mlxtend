# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from mlxtend.feature_selection import ColumnSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import datasets
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from packaging.version import Version
from sklearn import __version__ as sklearn_version


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
    pipe = make_pipeline(ColumnSelector(),
                         LogisticRegression(multi_class='ovr',
                                            solver='liblinear'))
    grid = {'columnselector__cols': [[1, 2], [1, 2, 3], 0, [1]],
            'logisticregression__C': [0.1, 1.0, 10.0]}

    if Version(sklearn_version) < Version("0.24.1"):
        gsearch1 = GridSearchCV(estimator=pipe,
                                param_grid=grid,
                                iid=False,
                                cv=5,
                                n_jobs=1,
                                refit=False)
    else:
        gsearch1 = GridSearchCV(estimator=pipe,
                                param_grid=grid,
                                cv=5,
                                n_jobs=1,
                                refit=False)

    gsearch1.fit(X, y)
    assert gsearch1.best_params_['columnselector__cols'] == [1, 2, 3]


def test_ColumnSelector_with_dataframe():
    boston = datasets.load_boston()
    df_in = pd.DataFrame(boston.data, columns=boston.feature_names)
    df_out = ColumnSelector(cols=('ZN', 'CRIM')).transform(df_in)
    assert df_out.shape == (506, 2)


def test_ColumnSelector_with_dataframe_and_int_columns():
    boston = datasets.load_boston()
    df_in = pd.DataFrame(boston.data, columns=boston.feature_names)
    df_out_str = ColumnSelector(cols=('INDUS', 'CHAS')).transform(df_in)
    df_out_int = ColumnSelector(cols=(2, 3)).transform(df_in)

    np.testing.assert_array_equal(df_out_str[:, 0],
                                  df_out_int[:, 0])
    np.testing.assert_array_equal(df_out_str[:, 1],
                                  df_out_int[:, 1])


def test_ColumnSelector_with_dataframe_drop_axis():
    boston = datasets.load_boston()
    df_in = pd.DataFrame(boston.data, columns=boston.feature_names)
    X1_out = ColumnSelector(cols='ZN', drop_axis=True).transform(df_in)
    assert X1_out.shape == (506,)

    X1_out = ColumnSelector(cols=('ZN',), drop_axis=True).transform(df_in)
    assert X1_out.shape == (506,)

    X1_out = ColumnSelector(cols='ZN').transform(df_in)
    assert X1_out.shape == (506, 1)

    X1_out = ColumnSelector(cols=('ZN',)).transform(df_in)
    assert X1_out.shape == (506, 1)


def test_ColumnSelector_with_dataframe_in_gridsearch():
    boston = datasets.load_boston()
    X = pd.DataFrame(boston.data, columns=boston.feature_names)
    y = boston.target
    pipe = make_pipeline(ColumnSelector(),
                         LinearRegression())
    grid = {'columnselector__cols': [
            ['ZN', 'RM'],
            ['ZN', 'RM', 'AGE'],
            'ZN', ['RM']
            ],
            'linearregression__copy_X': [True, False],
            'linearregression__fit_intercept': [True, False]
            }

    if Version(sklearn_version) < Version("0.24.1"):
        gsearch1 = GridSearchCV(estimator=pipe,
                                param_grid=grid,
                                cv=5,
                                n_jobs=1,
                                iid=False,
                                scoring='neg_mean_squared_error',
                                refit=False)
    else:
        gsearch1 = GridSearchCV(estimator=pipe,
                                param_grid=grid,
                                cv=5,
                                n_jobs=1,
                                scoring='neg_mean_squared_error',
                                refit=False)

    gsearch1.fit(X, y)
    assert gsearch1.best_params_['columnselector__cols'] == ['ZN', 'RM', 'AGE']
