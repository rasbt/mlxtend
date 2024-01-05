# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source Iris Flower dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os

import numpy as np

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "iris.csv.gz")


def iris_data(version="uci"):
    """Iris flower dataset.

    Source : https://archive.ics.uci.edu/ml/datasets/Iris
    Number of samples : 150
    Class labels : {0, 1, 2}, distribution: [50, 50, 50]
        0 = setosa, 1 = versicolor, 2 = virginica.

    Dataset Attributes:

        - 1) sepal length [cm]
        - 2) sepal width [cm]
        - 3) petal length [cm]
        - 4) petal width [cm]


    Parameters
    --------
    version : string, optional (default: 'uci').
      Version to use {'uci', 'corrected'}. 'uci' loads the dataset
      as deposited on the UCI machine learning repository, and
      'corrected' provides the version that is consistent with
      Fisher's original paper. See Note for details.


    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 150 flower samples as rows,
        and 4 feature columns sepal length, sepal width,
        petal length, and petal width.
        y is a 1-dimensional array of the class labels {0, 1, 2}


    Note
    --------
    The Iris dataset (originally collected by Edgar Anderson) and
    available in UCI's machine learning repository is different from
    the Iris dataset described in the original paper by  R.A. Fisher [1]).
    Precisely, there are two data points (row number
    34 and 37) in UCI's Machine Learning repository are different from the
    origianlly published Iris dataset. Also, the original version of the Iris
    Dataset, which can be loaded via `version='corrected'` is the same
    as the one in R.

    [1] . A. Fisher (1936). "The use of multiple measurements in taxonomic
    problems". Annals of Eugenics. 7 (2): 179–188

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/data/iris_data/

    """
    if version == "uci":
        tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
        X, y = tmp[:, :-1], tmp[:, -1]
        y = y.astype(int)
    elif version == "corrected":
        tmp = np.genfromtxt(fname=DATA_PATH, delimiter=",")
        X, y = tmp[:, :-1], tmp[:, -1]
        X[34] = [4.9, 3.1, 1.5, 0.2]
        X[37] = [4.9, 3.6, 1.4, 0.1]
        y = y.astype(int)
    else:
        raise ValueError("version must be 'uci' or 'corrected'.")
    return X, y
