# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from mlxtend.evaluate import bias_variance_decomp
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from mlxtend.data import iris_data
from mlxtend.data import boston_housing_data
from sklearn.model_selection import train_test_split


def test_01_loss_tree():

    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123,
                                                        shuffle=True,
                                                        stratify=y)

    tree = DecisionTreeClassifier(random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            tree, X_train, y_train, X_test, y_test,
            loss='0-1_loss',
            random_seed=123)

    assert round(avg_expected_loss, 3) == 0.062
    assert round(avg_bias, 3) == 0.022
    assert round(avg_var, 3) == 0.040


def test_01_loss_bagging():

    X, y = iris_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123,
                                                        shuffle=True,
                                                        stratify=y)

    tree = DecisionTreeClassifier(random_state=123)
    bag = BaggingClassifier(base_estimator=tree,
                            random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            bag, X_train, y_train, X_test, y_test,
            loss='0-1_loss',
            random_seed=123)

    assert round(avg_expected_loss, 3) == 0.048
    assert round(avg_bias, 3) == 0.022
    assert round(avg_var, 3) == 0.026


def test_mse_tree():

    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123,
                                                        shuffle=True)

    tree = DecisionTreeRegressor(random_state=123)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            tree, X_train, y_train, X_test, y_test,
            loss='mse',
            random_seed=123)

    assert round(avg_expected_loss, 3) == 31.756
    assert round(avg_bias, 3) == 13.856
    assert round(avg_var, 3) == 17.900


def test_mse_bagging():

    X, y = boston_housing_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=123,
                                                        shuffle=True)

    tree = DecisionTreeRegressor(random_state=123)
    bag = BaggingRegressor(base_estimator=tree,
                           n_estimators=100,
                           random_state=123)

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
            bag, X_train, y_train, X_test, y_test,
            loss='mse',
            random_seed=123)

    assert round(avg_expected_loss, 3) == 18.622
    assert round(avg_bias, 3) == 15.378
    assert round(avg_var, 3) == 3.244
