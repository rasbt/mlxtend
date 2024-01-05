# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Feature Importance Estimation Through Permutation
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC, SVR

from mlxtend.evaluate import feature_importance_permutation
from mlxtend.utils import assert_raises


def test_num_rounds_not_int():
    assert_raises(
        ValueError,
        "num_rounds must be an integer.",
        feature_importance_permutation,
        lambda x, y: (x, y),
        np.array([[1], [2], [3]]),
        np.array([1, 2, 3]),
        "accuracy",
        1.23,
    )


def test_num_rounds_negative_int():
    assert_raises(
        ValueError,
        "num_rounds must be greater than 1.",
        feature_importance_permutation,
        lambda x, y: (x, y),
        np.array([[1], [2], [3]]),
        np.array([1, 2, 3]),
        "accuracy",
        -1,
    )


def test_metric_wrong():
    assert_raises(
        ValueError,
        (
            'metric must be either "r2", "accuracy", or a '
            "function with signature "
            "func(y_true, y_pred)."
        ),
        feature_importance_permutation,
        lambda x, y: (x, y),
        np.array([[1], [2], [3]]),
        np.array([1, 2, 3]),
        "some-metric",
    )


def test_classification():
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    svm = SVC(C=1.0, kernel="rbf", random_state=0, gamma="auto")
    svm.fit(X_train, y_train)

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=svm.predict,
        X=X_test,
        y=y_test,
        metric="accuracy",
        num_rounds=1,
        seed=1,
    )

    assert imp_vals.shape == (X_train.shape[1],)
    assert imp_all.shape == (X_train.shape[1], 1)
    assert imp_vals[0] > 0.2
    assert imp_vals[1] > 0.2
    assert imp_vals[2] > 0.2
    assert sum(imp_vals[3:]) <= 0.02


def test_regression():
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=2,
        n_targets=1,
        random_state=123,
        shuffle=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    svm = SVR(kernel="rbf", gamma="auto")
    svm.fit(X_train, y_train)

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=svm.predict,
        X=X_test,
        y=y_test,
        metric="r2",
        num_rounds=1,
        seed=123,
    )

    assert imp_vals.shape == (X_train.shape[1],)
    assert imp_all.shape == (X_train.shape[1], 1)
    assert imp_vals[0] > 0.2
    assert imp_vals[1] > 0.2
    assert sum(imp_vals[3:]) <= 0.01


def test_regression_custom_r2():
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=2,
        n_targets=1,
        random_state=123,
        shuffle=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    svm = SVR(kernel="rbf", gamma="auto")
    svm.fit(X_train, y_train)

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=svm.predict,
        X=X_test,
        y=y_test,
        metric=r2_score,
        num_rounds=1,
        seed=123,
    )

    assert imp_vals.shape == (X_train.shape[1],)
    assert imp_all.shape == (X_train.shape[1], 1)
    assert imp_vals[0] > 0.2
    assert imp_vals[1] > 0.2
    assert sum(imp_vals[3:]) <= 0.01


def test_regression_custom_mse():
    X, y = make_regression(
        n_samples=1000,
        n_features=5,
        n_informative=2,
        n_targets=1,
        random_state=123,
        shuffle=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123
    )

    svm = SVR(kernel="rbf", gamma="auto")
    svm.fit(X_train, y_train)

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=svm.predict,
        X=X_test,
        y=y_test,
        metric=mean_squared_error,
        num_rounds=1,
        seed=123,
    )

    norm_imp_vals = imp_vals / np.abs(imp_vals).max()

    assert imp_vals.shape == (X_train.shape[1],)
    assert imp_all.shape == (X_train.shape[1], 1)
    assert norm_imp_vals[0] == -1.0


def test_n_rounds():
    X, y = make_classification(
        n_samples=1000,
        n_features=6,
        n_informative=3,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        random_state=0,
        shuffle=False,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    svm = SVC(C=1.0, kernel="rbf", random_state=0, gamma="auto")
    svm.fit(X_train, y_train)

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=svm.predict,
        X=X_test,
        y=y_test,
        metric="accuracy",
        num_rounds=100,
        seed=1,
    )

    assert imp_vals.shape == (X_train.shape[1],)
    assert imp_all.shape == (X_train.shape[1], 100)
    assert imp_vals[0].mean() > 0.2
    assert imp_vals[1].mean() > 0.2


def test_feature_groups():
    df_data = pd.read_csv(
        "https://gist.githubusercontent.com/rasbt/"
        "b99bf69079bc0d601eeae8a49248d358/"
        "raw/a114be9801647ec5460089f3a9576713dabf5f1f/"
        "onehot-numeric-mixed-data.csv"
    )

    df_X = df_data[["measurement1", "measurement2", "measurement3", "categorical"]]
    df_y = df_data["label"]

    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(
        df_X, df_y, test_size=0.33, random_state=42, stratify=df_y
    )

    ohe = OneHotEncoder(drop="first")
    ohe.fit(df_X_train[["categorical"]])

    df_X_train_ohe = df_X_train.drop(columns=["categorical"])
    df_X_test_ohe = df_X_test.drop(columns=["categorical"])

    ohe_train = np.asarray(ohe.transform(df_X_train[["categorical"]]).todense())
    ohe_test = np.asarray(ohe.transform(df_X_test[["categorical"]]).todense())

    X_train_ohe = np.hstack((df_X_train_ohe.values, ohe_train))
    X_test_ohe = np.hstack((df_X_test_ohe.values, ohe_test))

    model = LogisticRegression().fit(X_train_ohe, df_y_train.values)

    feature_groups = [0, 1, 2, range(3, 21)]

    imp_vals, imp_all = feature_importance_permutation(
        predict_method=model.predict,
        X=X_test_ohe,
        y=df_y_test.values,
        metric="accuracy",
        num_rounds=50,
        feature_groups=feature_groups,
        seed=1,
    )

    assert len(imp_vals) == 4
