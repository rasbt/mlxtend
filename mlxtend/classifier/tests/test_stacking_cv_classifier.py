# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Authors: Sebastian Raschka <sebastianraschka.com>
#          Reiichiro Nakano <github.com/reiinakano>
#
# License: BSD 3 clause

import random
import pandas as pd
import numpy as np
from scipy import sparse
from mlxtend.classifier import StackingCVClassifier
from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.utils import assert_raises
from mlxtend.data import iris_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from nose.tools import raises
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version


X_iris, y_iris = iris_data()
X_iris = X_iris[:, 1:3]

breast_cancer = datasets.load_breast_cancer()
X_breast, y_breast = breast_cancer.data[:, 1:3], breast_cancer.target


def test_StackingCVClassifier():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))

    if Version(sklearn_version) < Version("0.20"):
        assert scores_mean == 0.93, scores_mean
    else:
        assert scores_mean == 0.92, scores_mean


def test_sample_weight():
    # with no weight given
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)
    prob1 = sclf.fit(X_iris, y_iris).predict_proba(X_iris)

    # with weight = 1
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)
    w = np.ones(len(y_iris))
    prob2 = sclf.fit(X_iris, y_iris,
                     sample_weight=w).predict_proba(X_iris)

    # with random weight
    random.seed(87)
    w = np.array([random.random() for _ in range(len(y_iris))])
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)
    prob3 = sclf.fit(X_iris, y_iris,
                     sample_weight=w).predict_proba(X_iris)

    diff12 = np.max(np.abs(prob1 - prob2))
    diff23 = np.max(np.abs(prob2 - prob3))
    assert diff12 < 1e-3, "max diff is %.4f" % diff12
    assert diff23 > 1e-3, "max diff is %.4f" % diff23


@raises(TypeError)
def test_no_weight_support():
    w = np.array([random.random() for _ in range(len(y_iris))])
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    clf3 = KNeighborsClassifier()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                                meta_classifier=meta,
                                shuffle=False)
    sclf.fit(X_iris, y_iris, sample_weight=w)


@raises(TypeError)
def test_no_weight_support_meta():
    w = np.array([random.random() for _ in range(len(y_iris))])
    meta = KNeighborsClassifier()
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)
    sclf.fit(X_iris, y_iris, sample_weight=w)


def test_no_weight_support_with_no_weight():
    logit = LogisticRegression(multi_class='ovr', solver='liblinear')
    rf = RandomForestClassifier(n_estimators=10)
    gnb = GaussianNB()
    knn = KNeighborsClassifier()
    sclf = StackingCVClassifier(classifiers=[logit, rf, gnb],
                                meta_classifier=knn,
                                shuffle=False)
    sclf.fit(X_iris, y_iris)

    sclf = StackingCVClassifier(classifiers=[logit, knn, gnb],
                                meta_classifier=rf,
                                shuffle=False)
    sclf.fit(X_iris, y_iris)


def test_StackingClassifier_proba():

    np.random.seed(12)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))

    if Version(sklearn_version) < Version("0.20"):
        assert scores_mean == 0.92, scores_mean
    else:
        assert scores_mean == 0.93, scores_mean


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                use_probas=True,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    X, y = iris_data()
    grid.fit(X, y)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.96, 0.95, 0.96, 0.95]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20],
              'use_probas': [True, False]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5, iid=False)
    X, y = iris_data()
    grid = grid.fit(X, y)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94, scores_mean


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_features_in_secondary=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_do_not_stratify():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                stratify=False)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_cross_validation_technique():
    # This is like the `test_do_not_stratify` but instead
    # autogenerating the cross validation strategy it provides
    # a pre-created object
    np.random.seed(123)
    cv = KFold(n_splits=2, shuffle=True)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                cv=cv)

    scores = cross_val_score(sclf,
                             X_iris,
                             y_iris,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta, shuffle=False)

    X, y = iris_data()
    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  X)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  X)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_meta_features,
                  X)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=3)
    sclf.fit(X_iris, y_iris)


def test_list_of_lists():
    X_list = [i for i in X_iris]
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=0)

    try:
        sclf.fit(X_list, y_iris)
    except TypeError as e:
        assert 'are NumPy arrays. If X and y are lists' in str(e)


def test_pandas():
    X_df = pd.DataFrame(X_iris)
    meta = LogisticRegression(multi_class='ovr', solver='liblinear')
    clf1 = RandomForestClassifier(n_estimators=10)
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=0)
    try:
        sclf.fit(X_df, y_iris)
    except KeyError as e:
        assert 'are NumPy arrays. If X and y are pandas DataFrames' in str(e)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3],
                                meta_classifier=lr)

    got = sorted(list({s.split('__')[0] for s in sclf.get_params().keys()}))
    expect = ['classifiers',
              'cv',
              'gaussiannb',
              'kneighborsclassifier',
              'meta-logisticregression',
              'meta_classifier',
              'randomforestclassifier',
              'shuffle',
              'store_train_meta_features',
              'stratify',
              'use_clones',
              'use_features_in_secondary',
              'use_probas',
              'verbose']
    assert got == expect, got


def test_classifier_gridsearch():
    np.random.seed(123)
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(n_estimators=10)
    clf3 = GaussianNB()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    sclf = StackingCVClassifier(classifiers=[clf1],
                                meta_classifier=lr)

    params = {'classifiers': [[clf1], [clf1, clf2, clf3]]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        iid=False,
                        refit=True)
    grid.fit(X_iris, y_iris)

    assert len(grid.best_params_['classifiers']) == 3, \
        len(grid.best_params_['classifiers'])


def test_train_meta_features_():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    gnb = GaussianNB()
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 store_train_meta_features=True)
    X_train, X_test, y_train,  y_test = train_test_split(X_iris, y_iris,
                                                         test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    gnb = GaussianNB()
    X_train, X_test, y_train,  y_test = train_test_split(X_iris, y_iris,
                                                         test_size=0.3)
    #  test default (class labels)
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 store_train_meta_features=True)
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)


def test_meta_feat_reordering():
    np.random.seed(123)
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    gnb = GaussianNB()
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 shuffle=True,
                                 store_train_meta_features=True)
    X_train, X_test, y_train,  y_test = train_test_split(X_breast, y_breast,
                                                         test_size=0.3)
    stclf.fit(X_train, y_train)

    assert round(roc_auc_score(y_train,
                 stclf.train_meta_features_[:, 1]), 2) == 0.87, \
        round(roc_auc_score(y_train,
              stclf.train_meta_features_[:, 1]), 2)


def test_clone():
    knn = KNeighborsClassifier()
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    gnb = GaussianNB()
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 store_train_meta_features=True)
    clone(stclf)


def test_sparse_inputs():
    np.random.seed(123)
    rf = RandomForestClassifier(n_estimators=10)
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    stclf = StackingCVClassifier(classifiers=[rf, rf],
                                 meta_classifier=lr)
    X_train, X_test, y_train,  y_test = train_test_split(X_breast, y_breast,
                                                         test_size=0.3)

    # dense
    stclf.fit(X_train, y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99

    # sparse
    stclf.fit(sparse.csr_matrix(X_train), y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99


def test_sparse_inputs_with_features_in_secondary():
    np.random.seed(123)
    rf = RandomForestClassifier(n_estimators=10)
    lr = LogisticRegression(multi_class='ovr', solver='liblinear')
    stclf = StackingCVClassifier(classifiers=[rf, rf],
                                 meta_classifier=lr,
                                 use_features_in_secondary=True)
    X_train, X_test, y_train,  y_test = train_test_split(X_breast, y_breast,
                                                         test_size=0.3)

    # dense
    stclf.fit(X_train, y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99, \
        round(stclf.score(X_train, y_train), 2)

    # sparse
    stclf.fit(sparse.csr_matrix(X_train), y_train)
    assert round(stclf.score(X_train, y_train), 2) == 0.99, \
        round(stclf.score(X_train, y_train), 2)
