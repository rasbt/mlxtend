# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from distutils.version import LooseVersion as Version
from numpy.testing import assert_almost_equal
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.classifier import SoftmaxRegression
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from mlxtend.utils import assert_raises
from sklearn.model_selection import GroupKFold
from sklearn import __version__ as sklearn_version


def dict_compare_utility(d1, d2):
    assert d1.keys() == d2.keys(), "%s != %s" % (d1, d2)
    for i in d1:
        err_msg1 = ("d1[%s]['feature_idx']"
                    " != d2[%s]['feature_idx']" % (i, i))
        err_msg2 = ("d1[%s]['feature_names']"
                    " != d2[%s]['feature_names']" % (i, i))
        assert d1[i]['feature_idx'] == d2[i]["feature_idx"], err_msg1
        assert d1[i]['feature_names'] == d2[i]["feature_names"], err_msg2
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
                  'feature_names': ('0', '1'),
                  'avg_score': 0.82666666666666666,
                  'cv_scores': np.array([0.82666667])},
              1: {'feature_idx': (0, 2),
                  'feature_names': ('0', '2'),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              2: {'feature_idx': (0, 3),
                  'feature_names': ('0', '3'),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              3: {'feature_idx': (1, 2),
                  'feature_names': ('1', '2'),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              4: {'feature_idx': (1, 3),
                  'feature_names': ('1', '3'),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              5: {'feature_idx': (2, 3),
                  'feature_names': ('2', '3'),
                  'avg_score': 0.97333333333333338,
                  'cv_scores': np.array([0.97333333])},
              6: {'feature_idx': (0, 1, 2),
                  'feature_names': ('0', '1', '2'),
                  'avg_score': 0.95999999999999996,
                  'cv_scores': np.array([0.96])},
              7: {'feature_idx': (0, 1, 3),
                  'feature_names': ('0', '1', '3'),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              8: {'feature_idx': (0, 2, 3),
                  'feature_names': ('0', '2', '3'),
                  'avg_score': 0.96666666666666667,
                  'cv_scores': np.array([0.96666667])},
              9: {'feature_idx': (1, 2, 3),
                  'feature_names': ('1', '2', '3'),
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
    expect = {0: {'avg_score': 0.9329658605974395,
                  'feature_idx': (0, 1, 2),
                  'feature_names': ('0', '1', '2'),
                  'cv_scores': np.array([0.974, 0.947, 0.892, 0.919])},
              1: {'avg_score': 0.9400782361308677,
                  'feature_idx': (0, 1, 3),
                  'feature_names': ('0', '1', '3'),
                  'cv_scores': np.array([0.921, 0.947, 0.919, 0.973])},
              2: {'avg_score': 0.9532361308677098,
                  'feature_idx': (0, 2, 3),
                  'feature_names': ('0', '2', '3'),
                  'cv_scores': np.array([0.974, 0.947, 0.919, 0.973])},
              3: {'avg_score': 0.97275641025641035,
                  'feature_idx': (1, 2, 3),
                  'feature_names': ('1', '2', '3'),
                  'cv_scores': np.array([0.974, 1., 0.946, 0.973])}}

    if Version(sklearn_version) < Version("1.0"):
        expect[0]['avg_score'] = 0.9391025641025641
        expect[0]['cv_scores'] = np.array([0.974, 0.947, 0.892, 0.946])
        expect[2]['avg_score'] = 0.9529914529914529

    if Version(sklearn_version) < Version("0.22"):
        expect[0]['cv_scores'] = np.array([0.97435897, 0.94871795,
                                           0.88888889, 0.94444444])
        expect[1]['cv_scores'] = np.array([0.92307692, 0.94871795,
                                           0.91666667, 0.97222222])
        expect[2]['cv_scores'] = np.array([0.97435897, 0.94871795,
                                           0.91666667, 0.97222222])
        expect[3]['cv_scores'] = np.array([0.97435897, 0.94871795,
                                           0.91666667, 0.97222222])
        expect[1]['avg_score'] = 0.94017094017094016
        assert round(efs1.best_score_, 4) == 0.9728
    else:
        assert round(efs1.best_score_, 4) == 0.9732

    dict_compare_utility(d1=expect, d2=efs1.subsets_)
    assert efs1.best_idx_ == (1, 2, 3)
    assert efs1.best_feature_names_ == ('1', '2', '3')


def test_knn_cv3_groups():
    iris = load_iris()
    X = iris.data
    y = iris.target
    knn = KNeighborsClassifier(n_neighbors=4)
    efs1 = EFS(knn,
               min_features=3,
               max_features=3,
               scoring='accuracy',
               cv=GroupKFold(n_splits=3),
               print_progress=False)
    np.random.seed(1630672634)
    groups = np.random.randint(0, 6, size=len(y))
    efs1 = efs1.fit(X, y, groups=groups)

    expect = {0: {'cv_scores': np.array([0.97916667, 0.93877551, 0.9245283]),
                  'feature_idx': (0, 1, 2),
                  'avg_score': 0.9474901595858469,
                  'feature_names': ('0', '1', '2')},
              1: {'cv_scores': np.array([1., 0.93877551, 0.9245283]),
                  'feature_idx': (0, 1, 3),
                  'avg_score': 0.9544346040302915,
                  'feature_names': ('0', '1', '3')},
              2: {'cv_scores': np.array([0.97916667, 0.95918367, 0.9245283]),
                  'feature_idx': (0, 2, 3),
                  'avg_score': 0.9542928806742822,
                  'feature_names': ('0', '2', '3')},
              3: {'cv_scores': np.array([0.97916667, 0.95918367, 0.94339623]),
                  'feature_idx': (1, 2, 3),
                  'avg_score': 0.9605821888503829,
                  'feature_names': ('1', '2', '3')}}
    dict_compare_utility(d1=expect, d2=efs1.subsets_)


def test_fit_params():
    iris = load_iris()
    X = iris.data
    y = iris.target
    sample_weight = np.ones(X.shape[0])
    forest = RandomForestClassifier(n_estimators=100, random_state=123)
    efs1 = EFS(forest,
               min_features=3,
               max_features=3,
               scoring='accuracy',
               cv=4,
               print_progress=False)
    efs1 = efs1.fit(X, y, sample_weight=sample_weight)
    expect = {0: {'feature_idx': (0, 1, 2),
                  'feature_names': ('0', '1', '2'),
                  'cv_scores': np.array([0.947, 0.868, 0.919, 0.973]),
                  'avg_score': 0.9269203413940257},
              1: {'feature_idx': (0, 1, 3),
                  'feature_names': ('0', '1', '3'),
                  'cv_scores': np.array([0.921, 0.921, 0.892, 1.]),
                  'avg_score': 0.9337606837606838},
              2: {'feature_idx': (0, 2, 3),
                  'feature_names': ('0', '2', '3'),
                  'cv_scores': np.array([0.974, 0.947, 0.919, 0.973]),
                  'avg_score': 0.9532361308677098},
              3: {'feature_idx': (1, 2, 3),
                  'feature_names': ('1', '2', '3'),
                  'cv_scores': np.array([0.974, 0.947, 0.892, 1.]),
                  'avg_score': 0.9532361308677098}}

    if Version(sklearn_version) < Version("0.22"):
        expect[0]['avg_score'] = 0.9401709401709402
        expect[0]['cv_scores'] = np.array([0.94871795, 0.92307692,
                                           0.91666667, 0.97222222])
        expect[1]['cv_scores'] = np.array([0.94871795, 0.92307692,
                                           0.91666667, 0.97222222])
        expect[2]['cv_scores'] = np.array([0.94871795, 0.92307692,
                                           0.91666667, 0.97222222])
        expect[2]['avg_score'] = 0.9599358974358974
        expect[3]['avg_score'] = 0.9599358974358974
        expect[3]['cv_scores'] = np.array([0.97435897, 0.94871795,
                                           0.91666667, 1.])
        assert round(efs1.best_score_, 4) == 0.9599

    else:
        assert round(efs1.best_score_, 4) == 0.9532

    dict_compare_utility(d1=expect, d2=efs1.subsets_)
    assert efs1.best_idx_ == (0, 2, 3)


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
    class Perceptron(object):

        def __init__(self, eta=0.1, epochs=50, random_seed=None,
                     print_progress=0):

            self.eta = eta
            self.epochs = epochs
            self.random_seed = random_seed
            self.print_progress = print_progress
            self._is_fitted = False

        def _fit(self, X, y, init_params=True):
            self._check_target_array(y, allowed={(0, 1)})
            y_data = np.where(y == 0, -1., 1.)

            if init_params:
                self.b_, self.w_ = self._init_params(
                    weights_shape=(X.shape[1], 1),
                    bias_shape=(1,),
                    random_seed=self.random_seed)
                self.cost_ = []

            rgen = np.random.RandomState(self.random_seed)
            for i in range(self.epochs):
                errors = 0

                for idx in self._yield_minibatches_idx(
                        rgen=rgen,
                        n_batches=y_data.shape[0],
                        data_ary=y_data, shuffle=True):

                    update = self.eta * (y_data[idx] -
                                         self._to_classlabels(X[idx]))
                    self.w_ += (update * X[idx]).reshape(self.w_.shape)
                    self.b_ += update
                    errors += int(update != 0.0)

                if self.print_progress:
                    self._print_progress(iteration=i + 1,
                                         n_iter=self.epochs,
                                         cost=errors)
                self.cost_.append(errors)
            return self

        def _net_input(self, X):
            """ Net input function """
            return (np.dot(X, self.w_) + self.b_).flatten()

        def _to_classlabels(self, X):
            return np.where(self._net_input(X) < 0.0, -1., 1.)

        def _predict(self, X):
            return np.where(self._net_input(X) < 0.0, 0, 1)

    expect = ("Cannot clone object. You should provide an"
              " instance of scikit-learn estimator instead of a class.")

    assert_raises(TypeError,
                  expect,
                  EFS,
                  Perceptron,
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


def test_custom_feature_names():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X = iris.data
    y = iris.target
    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    efs1 = efs1.fit(X, y, custom_feature_names=(
          'sepal length', 'sepal width', 'petal length', 'petal width'))
    assert efs1.best_idx_ == (2, 3), efs1.best_idx_
    assert efs1.best_feature_names_ == ('petal length', 'petal width')


def test_check_pandas_dataframe_fit():

    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X = iris.data
    y = iris.target
    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    df = pd.DataFrame(X, columns=['sepal length', 'sepal width',
                                  'petal length', 'petal width'])

    sfs1 = efs1.fit(X, y)
    assert efs1.best_idx_ == (2, 3), efs1.best_idx_
    assert efs1.best_feature_names_ == ('2', '3')
    assert efs1.interrupted_ is False

    sfs1._TESTING_INTERRUPT_MODE = True
    sfs1 = sfs1.fit(df, y)
    assert efs1.best_idx_ == (0, 1), efs1.best_idx_
    assert efs1.best_feature_names_ == ('sepal length', 'sepal width')
    assert efs1.interrupted_ is True


def test_check_pandas_dataframe_transform():
    knn = KNeighborsClassifier(n_neighbors=4)
    iris = load_iris()
    X = iris.data
    y = iris.target
    efs1 = EFS(knn,
               min_features=2,
               max_features=2,
               scoring='accuracy',
               cv=0,
               clone_estimator=False,
               print_progress=False,
               n_jobs=1)

    df = pd.DataFrame(X, columns=['sepal length', 'sepal width',
                                  'petal length', 'petal width'])
    efs1 = efs1.fit(df, y)
    assert efs1.best_idx_ == (2, 3)
    assert (150, 2) == efs1.transform(df).shape
