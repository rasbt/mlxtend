# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source Boston Housing dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "boston_housing.csv")


def boston_housing_data():
    """Boston Housing dataset.

    Source : https://archive.ics.uci.edu/ml/datasets/Housing
    Number of samples : 506

    Continuous target variable : MEDV
    MEDV = Median value of owner-occupied homes in $1000's

    Dataset Attributes:

        - 1) CRIM      per capita crime rate by town
        - 2) ZN        proportion of residential land zoned for lots over
                 25,000 sq.ft.
        - 3) INDUS     proportion of non-retail business acres per town
        - 4) CHAS      Charles River dummy variable (= 1 if tract bounds
                 river; 0 otherwise)
        - 5) NOX       nitric oxides concentration (parts per 10 million)
        - 6) RM        average number of rooms per dwelling
        - 7) AGE       proportion of owner-occupied units built prior to 1940
        - 8) DIS       weighted distances to five Boston employment centres
        - 9) RAD       index of accessibility to radial highways
        - 10) TAX      full-value property-tax rate per $10,000
        - 11) PTRATIO  pupil-teacher ratio by town
        - 12) B        1000(Bk - 0.63)^2 where Bk is the prop. of b. by town
        - 13) LSTAT    % lower status of the population

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 506 housing samples as rows
        and 13 feature columns.
        y is a 1-dimensional array of the continuous target variable MEDV

    """
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',', dtype=float)
    print(tmp.shape)
    X, y = tmp[:, :-1], tmp[:, -1]
    return X, y
