# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.model_selection import train_test_split

from mlxtend.classifier import OneRClassifier
from mlxtend.data import iris_data

X, y = iris_data()


def get_feature_quartiles(X):
    X_discretized = X.copy()
    for col in range(X.shape[1]):
        for q, class_label in zip([1.0, 0.75, 0.5, 0.25], [3, 2, 1, 0]):
            threshold = np.quantile(X[:, col], q=q)
            X_discretized[X[:, col] <= threshold, col] = class_label
    return X_discretized.astype(np.int_)


Xd = get_feature_quartiles(X)
Xd_train, Xd_test, y_train, y_test = train_test_split(Xd, y, random_state=0, stratify=y)


def test_iris_quartiles_resolve_ties_first():
    oner = OneRClassifier()
    oner.fit(Xd_train, y_train)
    assert oner.feature_idx_ == 2
    assert oner.prediction_dict_["total error"] == 16
    assert round(oner.score(Xd_train, y_train), 4) == 0.8571
    assert round(oner.score(Xd_test, y_test), 4) == 0.8421


def test_iris_quartiles_resolve_ties_chi_squared_1():
    oner = OneRClassifier(resolve_ties="chi-squared")
    oner.fit(Xd_train, y_train)
    assert oner.feature_idx_ == 2
    assert oner.prediction_dict_["total error"] == 16
    assert round(oner.score(Xd_train, y_train), 4) == 0.8571
    assert round(oner.score(Xd_test, y_test), 4) == 0.8421
    np.testing.assert_almost_equal(oner.p_value_, 0.0, decimal=7)


def test_iris_quartiles_resolve_ties_chi_squared_2():
    # tests with duplicate column
    oner = OneRClassifier(resolve_ties="chi-squared")

    Xd_traintemp = np.zeros((Xd_train.shape[0], Xd_train.shape[1] + 1))
    Xd_traintemp[:, 0] = Xd_train[:, 2]
    Xd_traintemp[:, 1] = Xd_train[:, 0]
    Xd_traintemp[:, 2] = Xd_train[:, 1]
    Xd_traintemp[:, 3] = Xd_train[:, 2]
    Xd_traintemp[:, 4] = Xd_train[:, 3]

    oner.fit(Xd_traintemp, y_train)
    assert oner.feature_idx_ == 0
    assert oner.prediction_dict_["total error"] == 16
    assert round(oner.score(Xd_traintemp, y_train), 4) == 0.8571
