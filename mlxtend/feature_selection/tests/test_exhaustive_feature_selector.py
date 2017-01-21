# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import sys
import numpy as np
from numpy.testing import assert_almost_equal
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import SoftmaxRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from mlxtend.utils import assert_raises
from mlxtend.classifier import SoftmaxRegression


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


def test_minfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn,
              min_features=0,
              max_features=2)
    expect = ('min_features must be smaller than 5 and larger than 0')
    assert_raises(AttributeError,
                  expect,
                  efs.fit,
                  X,
                  y)


def test_maxfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn,
              min_features=1,
              max_features=0)
    expect = ('max_features must be smaller than 5 and larger than 0')
    assert_raises(AttributeError,
                  expect,
                  efs.fit,
                  X,
                  y)


def test_minmaxfeatures_1():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier()

    efs = EFS(estimator=knn,
              min_features=3,
              max_features=2)
    expect = ('min_features must be <= max_features')
    assert_raises(AttributeError,
                  expect,
                  efs.fit,
                  X,
                  y)


def test_knn_wo_cv():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(knn,
               min_features=2,
               max_features=3,
               scoring='accuracy',
               cv=0,
               print_progress=False)
    efs1 = efs1.fit(X, y)
    expect = {0: {'feature_idx': (0, 1),
                  'avg_score': 0.82666666666666666,
                  'cv_scores': np.array([0.82666667])},
              1: {'feature_idx': (0, 2),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              2: {'feature_idx': (0, 3),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              3: {'feature_idx': (1, 2),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              4: {'feature_idx': (1, 3),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              5: {'feature_idx': (2, 3),
                  'avg_score': 0.97333333333333338,
                  'cv_scores': np.array([0.97333333])},
              6: {'feature_idx': (0, 1, 2),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              7: {'feature_idx': (0, 1, 3),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              8: {'feature_idx': (0, 2, 3),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              9: {'feature_idx': (1, 2, 3),
                  'avg_score': 0.97333333333333338,
                  'cv_scores': np.array([0.97333333])}}
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_knn_cv3():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(knn,
               min_features=3,
               max_features=3,
               scoring='accuracy',
               cv=4,
               print_progress=False)
    efs1 = efs1.fit(X, y)
    expect = {0: {'avg_score': 0.9391025641025641,
                  'feature_idx': (0, 1, 2),
                  'cv_scores': np.array([0.97435897, 0.94871795,
                                         0.88888889, 0.94444444])},
              1: {'avg_score': 0.94017094017094016,
                  'feature_idx': (0, 1, 3),
                  'cv_scores': np.array([0.92307692, 0.94871795,
                                         0.91666667, 0.97222222])},
              2: {'avg_score': 0.95299145299145294,
                  'feature_idx': (0, 2, 3),
                  'cv_scores': np.array([0.97435897, 0.94871795,
                                         0.91666667, 0.97222222])},
              3: {'avg_score': 0.97275641025641035,
                  'feature_idx': (1, 2, 3),
                  'cv_scores': np.array([0.97435897, 1.,
                                         0.94444444, 0.97222222])}}
    dict_compare_utility(d1=expect, d2=efs1.subsets_)
    assert efs1.best_idx_ == (1, 2, 3)
    assert round(efs1.best_score_, 4) == 0.9728


def test_regression():
    boston = load_boston()
    X, y = boston.data[:, [1, 2, 6, 8, 12]], boston.target
    lr = LinearRegression()
    efs_r = EFS(lr,
                min_features=3,
                max_features=4,
                scoring='neg_mean_squared_error',
                cv=10,
                print_progress=False)
    efs_r = efs_r.fit(X, y)
    assert efs_r.best_idx_ == (0, 2, 4)
    assert round(efs_r.best_score_, 4) == -40.8777


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
                  EFS,
                  SoftmaxRegression,
                  min_features=2,
                  max_features=2,
                  clone_estimator=True)


def test_clone_params_pass():
    iris = load_iris()
    X = iris.data
    y = iris.target
    lr = SoftmaxRegression(random_seed=1)
    efs1 = EFS(lr,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)
    efs1 = efs1.fit(X, y)
    assert(efs1.best_idx_ == (1, 3))


def test_transform_not_fitted():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    expect = 'ExhaustiveFeatureSelector has not been fitted, yet.'

    assert_raises(AttributeError,
                  expect,
                  efs1.transform,
                  X)


def test_fit_transform():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    X_t = efs1.fit_transform(X, y)
    assert X_t.shape == (150, 2)


def test_get_metric_dict_not_fitted():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)

    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    expect = 'ExhaustiveFeatureSelector has not been fitted, yet.'

    assert_raises(AttributeError,
                  expect,
                  efs1.get_metric_dict)
