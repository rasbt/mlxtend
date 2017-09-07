# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys

import numpy as np
from numpy import nan
from numpy.testing import assert_almost_equal
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from mlxtend.classifier import SoftmaxRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.utils import assert_raises


def nan_roc_auc_score(y_true, y_score, average='macro', sample_weight=None):
    if len(np.unique(y_true)) != 2:
        return np.nan
    else:
        return roc_auc_score(y_true, y_score,
                             average=average, sample_weight=sample_weight)


def test_knn_option_sfbs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 3),
               forward=False,
               floating=True,
               cv=4,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_max_feature_subset_size_in_tuple_range():
    boston = load_boston()
    X, y = boston.data, boston.target

    lr = LinearRegression()

    sfs = SFS(lr,
              k_features=(1, 5),
              forward=False,
              floating=True,
              scoring='neg_mean_squared_error',
              cv=10)

    sfs = sfs.fit(X, y)
    assert len(sfs.k_feature_idx_) == 5

#test_knn_option_sfbs_tuplerange_1()
test_max_feature_subset_size_in_tuple_range()