# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys
import numpy as np
from numpy.testing import assert_almost_equal
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import SoftmaxRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from mlxtend.utils import assert_raises


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


def test_run_default():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    sfs = SFS(estimator=knn,
              verbose=0)
    sfs.fit(X, y)
    assert sfs.k_feature_idx_ == (3, )


def test_kfeatures_type_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = ('k_features must be a positive integer between 1 and X.shape[1],'
              ' got 0')
    sfs = SFS(estimator=knn,
              verbose=0,
              k_features=0)
    assert_raises(AttributeError,
                  expect,
                  sfs.fit,
                  X,
                  y)


def test_kfeatures_type_2():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = 'k_features must be a positive integer or tuple'
    sfs = SFS(estimator=knn,
              verbose=0,
              k_features='abc')
    assert_raises(AttributeError,
                  expect,
                  sfs.fit,
                  X,
                  y)


def test_kfeatures_type_3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = ('k_features tuple min value must be in range(1, X.shape[1]+1).')
    sfs = SFS(estimator=knn,
              verbose=0,
              k_features=(0, 5))
    assert_raises(AttributeError,
                  expect,
                  sfs.fit,
                  X,
                  y)


def test_kfeatures_type_4():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = ('k_features tuple max value must be in range(1, X.shape[1]+1).')
    sfs = SFS(estimator=knn,
              verbose=0,
              k_features=(1, 5))
    assert_raises(AttributeError,
                  expect,
                  sfs.fit,
                  X,
                  y)


def test_kfeatures_type_5():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()
    expect = ('he min k_features value must be'
              ' larger than the max k_features value.')
    sfs = SFS(estimator=knn,
              verbose=0,
              k_features=(3, 1))
    assert_raises(AttributeError,
                  expect,
                  sfs.fit,
                  X,
                  y)


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
               verbose=0)
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
               verbose=0)
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
               verbose=0)
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
               verbose=0)
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
               verbose=0)
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
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert sfs4.k_feature_idx_ == (1, 2, 3)


def test_knn_option_sfbs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 3),
               forward=False,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sfbs_tuplerange_2():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 4),
               forward=False,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sffs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 3),
               forward=True,
               floating=True,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sfs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 3),
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


def test_knn_option_sbs_tuplerange_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=3)
    sfs4 = SFS(knn,
               k_features=(1, 3),
               forward=False,
               floating=False,
               scoring='accuracy',
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs4 = sfs4.fit(X, y)
    assert round(sfs4.k_score_, 3) == 0.967, sfs4.k_score_
    assert sfs4.k_feature_idx_ == (0, 2, 3), sfs4.k_feature_idx_


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
               verbose=0)
    sfs5 = sfs5.fit(X, y)
    assert round(sfs5.k_score_, 4) == 0.9728

    sfs6 = SFS(knn,
               k_features=3,
               forward=False,
               floating=True,
               cv=4,
               skip_if_stuck=True,
               verbose=0)
    sfs6 = sfs6.fit(X, y)
    assert round(sfs6.k_score_, 4) == 0.9728

    sfs7 = SFS(knn,
               k_features=3,
               forward=False,
               floating=True,
               scoring='f1_macro',
               cv=4,
               skip_if_stuck=True)
    sfs7 = sfs7.fit(X, y)
    assert round(sfs7.k_score_, 4) == 0.9727, sfs7.k_score_


def test_regression():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()
    sfs_r = SFS(lr,
                k_features=13,
                forward=True,
                floating=False,
                scoring='neg_mean_squared_error',
                cv=10,
                skip_if_stuck=True,
                verbose=0)
    sfs_r = sfs_r.fit(X, y)
    assert len(sfs_r.k_feature_idx_) == 13
    assert round(sfs_r.k_score_, 4) == -34.7631


def test_regression_in_range():
    boston = load_boston()
    X, y = boston.data, boston.target
    lr = LinearRegression()
    sfs_r = SFS(lr,
                k_features=(1, 13),
                forward=True,
                floating=False,
                scoring='neg_mean_squared_error',
                cv=10,
                skip_if_stuck=True,
                verbose=0)
    sfs_r = sfs_r.fit(X, y)
    assert len(sfs_r.k_feature_idx_) == 9
    assert round(sfs_r.k_score_, 4) == -31.1537


def test_clone_params_fail():
    iris = load_iris()
    X = iris.data
    y = iris.target

    if sys.version_info >= (3, 0):
        objtype = 'class'
    else:
        objtype = 'type'

    expect = ("Cannot clone object"
              " '<class 'mlxtend.classifier."
              "softmax_regression.SoftmaxRegression'>'"
              " (type <%s 'type'>): it does not seem to be a"
              " scikit-learn estimator as it does not"
              " implement a 'get_params' methods.") % objtype

    assert_raises(TypeError,
                  expect,
                  SFS,
                  SoftmaxRegression,
                  k_features=3,
                  clone_estimator=True)


def test_clone_params_pass():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)
    sfs1 = SFS(lr,
               k_features=2,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=0,
               skip_if_stuck=True,
               clone_estimator=False,
               verbose=0,
               n_jobs=1)
    sfs1 = sfs1.fit(X, y)
    assert(sfs1.k_feature_idx_ == (1, 3))


def test_transform_not_fitted():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(knn,
               k_features=2,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=0,
               skip_if_stuck=True,
               clone_estimator=False,
               verbose=0,
               n_jobs=1)

    expect = 'SequentialFeatureSelector has not been fitted, yet.'

    assert_raises(AttributeError,
                  expect,
                  sfs1.transform,
                  X)


def test_get_metric_dict_not_fitted():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(knn,
               k_features=2,
               forward=True,
               floating=False,
               scoring='accuracy',
               cv=0,
               skip_if_stuck=True,
               clone_estimator=False,
               verbose=0,
               n_jobs=1)

    expect = 'SequentialFeatureSelector has not been fitted, yet.'

    assert_raises(AttributeError,
                  expect,
                  sfs1.get_metric_dict)


def test_keyboard_interrupt():
    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)
    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        scoring='accuracy',
        cv=3,
        skip_if_stuck=True,
        clone_estimator=False,
        verbose=5,
        n_jobs=1
    )

    sfs1._TESTING_INTERRUPT_MODE = True
    out = sfs1.fit(X, y)

    assert len(out.subsets_.keys()) > 0
    assert sfs1.interrupted_
