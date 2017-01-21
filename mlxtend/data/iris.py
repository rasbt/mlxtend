# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source Iris Flower dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "iris.csv.gz")


def iris_data():
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

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 150 flower samples as rows,
        and 4 feature columns sepal length, sepal width,
        petal length, and petal width.
        y is a 1-dimensional array of the class labels {0, 1, 2}

    """
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
    X, y = tmp[:, :-1], tmp[:, -1]
    y = y.astype(int)
    return X, y

    return X, y
