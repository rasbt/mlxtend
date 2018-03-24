# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Authors: Sebastian Raschka <sebastianraschka.com>
#          Reiichiro Nakano <github.com/reiinakano>
#
# License: BSD 3 clause

import pandas as pd
import numpy as np
from mlxtend.classifier import StackingCVClassifier
from mlxtend.externals.estimator_checks import NotFittedError
from mlxtend.utils import assert_raises
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target


def test_StackingClassifier():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93


def test_StackingClassifier_proba():

    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93


def test_gridsearch():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                use_probas=True,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)

    mean_scores = [round(s, 2) for s
                   in grid.cv_results_['mean_test_score']]

    assert mean_scores == [0.96, 0.95, 0.96, 0.95]


def test_gridsearch_enumerate_names():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf1, clf2],
                                meta_classifier=meta,
                                shuffle=False)

    params = {'meta-logisticregression__C': [1.0, 100.0],
              'randomforestclassifier-1__n_estimators': [5, 10],
              'randomforestclassifier-2__n_estimators': [5, 20],
              'use_probas': [True, False]}

    grid = GridSearchCV(estimator=sclf, param_grid=params, cv=5)
    grid = grid.fit(iris.data, iris.target)


def test_use_probas():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94, scores_mean


def test_use_features_in_secondary():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_features_in_secondary=True,
                                meta_classifier=meta,
                                shuffle=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.93, scores_mean


def test_do_not_stratify():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                stratify=False)

    scores = cross_val_score(sclf,
                             X,
                             y,
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
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                meta_classifier=meta,
                                cv=cv)

    scores = cross_val_score(sclf,
                             X,
                             y,
                             cv=5,
                             scoring='accuracy')
    scores_mean = (round(scores.mean(), 2))
    assert scores_mean == 0.94


def test_not_fitted():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta, shuffle=False)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict,
                  iris.data)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_proba,
                  iris.data)

    assert_raises(NotFittedError,
                  "This StackingCVClassifier instance is not fitted yet."
                  " Call 'fit' with appropriate arguments"
                  " before using this method.",
                  sclf.predict_meta_features,
                  iris.data)


def test_verbose():
    np.random.seed(123)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=3)
    sclf.fit(iris.data, iris.target)


def test_list_of_lists():
    X_list = [i for i in X]
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=0)

    try:
        sclf.fit(X_list, iris.target)
    except TypeError as e:
        assert 'are NumPy arrays. If X and y are lists' in str(e)


def test_pandas():
    X_df = pd.DataFrame(X)
    meta = LogisticRegression()
    clf1 = RandomForestClassifier()
    clf2 = GaussianNB()
    sclf = StackingCVClassifier(classifiers=[clf1, clf2],
                                use_probas=True,
                                meta_classifier=meta,
                                shuffle=False,
                                verbose=0)
    try:
        sclf.fit(X_df, iris.target)
    except KeyError as e:
        assert 'are NumPy arrays. If X and y are pandas DataFrames' in str(e)


def test_get_params():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
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
              'refit',
              'shuffle',
              'store_train_meta_features',
              'stratify',
              'use_features_in_secondary',
              'use_probas',
              'verbose']
    assert got == expect, got


def test_classifier_gridsearch():
    clf1 = KNeighborsClassifier(n_neighbors=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    lr = LogisticRegression()
    sclf = StackingCVClassifier(classifiers=[clf1],
                                meta_classifier=lr)

    params = {'classifiers': [[clf1], [clf1, clf2, clf3]]}

    grid = GridSearchCV(estimator=sclf,
                        param_grid=params,
                        cv=5,
                        refit=True)
    grid.fit(X, y)

    assert len(grid.best_params_['classifiers']) == 3


def test_train_meta_features_():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 store_train_meta_features=True)
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3)
    stclf.fit(X_train, y_train)
    train_meta_features = stclf.train_meta_features_
    assert train_meta_features.shape == (X_train.shape[0], 2)


def test_predict_meta_features():
    knn = KNeighborsClassifier()
    lr = LogisticRegression()
    gnb = GaussianNB()
    X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size=0.3)

    #  test default (class labels)
    stclf = StackingCVClassifier(classifiers=[knn, gnb],
                                 meta_classifier=lr,
                                 store_train_meta_features=True)
    stclf.fit(X_train, y_train)
    test_meta_features = stclf.predict(X_test)
    assert test_meta_features.shape == (X_test.shape[0],)
