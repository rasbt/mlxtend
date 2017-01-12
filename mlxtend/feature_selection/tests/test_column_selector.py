# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from mlxtend.feature_selection import ColumnSelector
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV


def test_ColumnSelector():
    X1_in = np.ones((4, 8))
    X1_out = ColumnSelector(cols=(1, 3)).transform(X1_in)
    assert(X1_out.shape == (4, 2))


def test_ColumnSelector_in_gridsearch():
    iris = load_iris()
    X, y = iris.data, iris.target
    pipe = make_pipeline(ColumnSelector(),
                         LogisticRegression())
    grid = {'columnselector__cols': [[1, 2], [1, 2, 3], 0, [1]],
            'logisticregression__C': [0.1, 1.0, 10.0]}

    gsearch1 = GridSearchCV(estimator=pipe,
                            param_grid=grid,
                            cv=5,
                            n_jobs=1,
                            refit=False)

    gsearch1.fit(X, y)
    assert gsearch1.best_params_['columnselector__cols'] == [1, 2, 3]
