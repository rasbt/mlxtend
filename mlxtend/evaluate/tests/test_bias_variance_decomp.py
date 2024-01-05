# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import os

import pandas as pd
import pytest
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import bias_variance_decomp


def pandas_input_fail():
    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y
    )

    X_train = pd.DataFrame(X_train)

    tree = DecisionTreeClassifier(random_state=123)

    with pytest.raises(ValueError):
        avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            tree, X_train, y_train, X_test, y_test, loss="0-1_loss", random_seed=123
        )


def test_01_loss_tree():
    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y
    )

    tree = DecisionTreeClassifier(random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        tree, X_train, y_train, X_test, y_test, loss="0-1_loss", random_seed=123
    )

    assert round(avg_expected_loss, 3) == 0.062
    assert round(avg_bias, 3) == 0.022
    assert round(avg_var, 3) == 0.040


def test_01_loss_bagging():
    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True, stratify=y
    )

    tree = DecisionTreeClassifier(random_state=123)
    bag = BaggingClassifier(base_estimator=tree, random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        bag, X_train, y_train, X_test, y_test, loss="0-1_loss", random_seed=123
    )

    assert round(avg_expected_loss, 3) == 0.048
    assert round(avg_bias, 3) == 0.022
    assert round(avg_var, 3) == 0.026


def test_mse_tree():
    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True
    )

    tree = DecisionTreeRegressor(random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        tree, X_train, y_train, X_test, y_test, loss="mse", random_seed=123
    )

    assert round(avg_expected_loss, 3) == 31.536
    assert round(avg_bias, 3) == 14.096
    assert round(avg_var, 3) == 17.440


def test_mse_bagging():
    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True
    )

    tree = DecisionTreeRegressor(random_state=123)
    bag = BaggingRegressor(base_estimator=tree, n_estimators=10, random_state=123)

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        bag, X_train, y_train, X_test, y_test, loss="mse", random_seed=123
    )

    assert round(avg_expected_loss, 2) == 20.24, avg_expected_loss
    assert round(avg_bias, 2) == 15.63, avg_bias
    assert round(avg_var, 2) == 4.61, avg_var


if "TRAVIS" in os.environ or os.environ.get("TRAVIS") == "true":
    TRAVIS = True
else:
    TRAVIS = False

if "APPVEYOR" in os.environ or os.environ.get("APPVEYOR") == "true":
    APPVEYOR = True
else:
    APPVEYOR = False


@pytest.mark.skipif(TRAVIS or APPVEYOR, reason="TensorFlow dependency")
def test_keras():
    import tensorflow as tf

    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=123, shuffle=True
    )

    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(32, activation=tf.nn.relu), tf.keras.layers.Dense(1)]
    )

    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    model.fit(X_train, y_train, epochs=10)
    model.predict(X_test)
