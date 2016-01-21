# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from numpy.testing import assert_almost_equal
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston


def dict_compare_utility(d1, d2):
    assert d1.keys() == d2.keys(), "%s != %s" % (d1, d2)
    for i in d1:
        err_msg = ("d1[%s]['feature_idx']"
                   " != d2[%s]['feature_idx']" % (i, i))
        assert d1[i]['feature_idx'] == d1[i]["feature_idx"], err_msg
        assert_almost_equal(d1[i]['avg_score'],
                            d2[i]['avg_score'],
                            decimal=3,
                            err_msg=("d1[%s]['avg_score']"
                                     " != d2[%s]['avg_score']" % (i, i)))
        assert_almost_equal(d1[i]['cv_scores'],
                            d2[i]['cv_scores'],
                            decimal=3,
                            err_msg=("d1[%s]['cv_scores']"
                                     " != d2[%s]['cv_scores']" % (i, i)))


def test_knn_wo_cv():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn,
               k_features=3,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=0,
               skip_if_stuck=True,
               print_progress=False)
    sfs1 = sfs1.fit(X, y)
    expect = {1: {'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96]),
                  'feature_idx': (3,)},
              2: {'avg_score': 0.97333333333333338,
                  'cv_scores': np.array([0.97333333]),
                  'feature_idx': (2, 3)},
              3: {'avg_score': 0.97333333333333338,
                  'cv_scores': np.array([0.97333333]),
                  'feature_idx': (1, 2, 3)}}
    dict_compare_utility(d1=expect, d2=sfs1.subsets_)


def test_knn_cv3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn,
               k_features=3,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs1 = sfs1.fit(X, y)
    sfs1.subsets_
    expect = {1: {'avg_score': 0.95299145299145294,
                  'cv_scores': np.array([0.97435897,
                                         0.94871795,
                                         0.88888889,
                                         1.0]),
                  'feature_idx': (3,)},
              2: {'avg_score': 0.95993589743589736,
                  'cv_scores': np.array([0.97435897,
                                         0.94871795,
                                         0.91666667,
                                         1.0]),
                  'feature_idx': (2, 3)},
              3: {'avg_score': 0.97275641025641035,
                  'cv_scores': np.array([0.97435897,
                                         1.0,
                                         0.94444444,
                                         0.97222222]),
                  'feature_idx': (1, 2, 3)}}
    dict_compare_utility(d1=expect, d2=sfs1.subsets_)


def test_knn_option_sfs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(knn,
               k_features=3,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs1 = sfs1.fit(X, y)
    assert sfs1.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sffs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs2 = SFS(knn,
               k_features=3,
               forward=True,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs2 = sfs2.fit(X, y)
    assert sfs2.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sbs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs3 = SFS(knn,
               k_features=3,
               forward=False,
               floating=False,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs3 = sfs3.fit(X, y)
    assert sfs3.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sfbs():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs4 = SFS(knn,
               k_features=3,
               forward=False,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs4 = sfs4.fit(X, y)
    assert sfs4.k_feature_idx_ == (1, 2, 3)


def test_knn_scoring_metric():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    sfs5 = SFS(knn,
               k_features=3,
               forward=False,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs5 = sfs5.fit(X, y)
    assert round(sfs5.k_score_, 4) == 0.9728

    sfs6 = SFS(knn,
               k_features=3,
               forward=False,
               floating=True,
               scoring='precision',
               cv=4,
               skip_if_stuck=True,
               print_progress=False)
    sfs6 = sfs6.fit(X, y)
    assert round(sfs6.k_score_, 4) == 0.9737


def test_regression():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()
    sfs_r = SFS(lr,
                k_features=13,
                forward=True,
                floating=False,
                scoring='mean_squared_error',
                cv=10,
                skip_if_stuck=True,
                print_progress=False)
    sfs_r = sfs_r.fit(X, y)
    assert round(sfs_r.k_score_, 4) == -34.7631
