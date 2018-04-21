# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source Wine dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "wine.csv")


def wine_data():
    """Wine dataset.

    Source : https://archive.ics.uci.edu/ml/datasets/Wine
    Number of samples : 178
    Class labels : {0, 1, 2}, distribution: [59, 71, 48]

    Dataset Attributes:

        - 1) Alcohol
        - 2) Malic acid
        - 3) Ash
        - 4) Alcalinity of ash
        - 5) Magnesium
        - 6) Total phenols
        - 7) Flavanoids
        - 8) Nonflavanoid phenols
        - 9) Proanthocyanins
        - 10) Color intensity
        - 11) Hue
        - 12) OD280/OD315 of diluted wines
        - 13) Proline

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 178 wine samples as rows
        and 13 feature columns.
        y is a 1-dimensional array of the 3 class labels 0, 1, 2

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/wine_data

    """
    tmp = np.loadtxt(DATA_PATH, delimiter=',')
    X, y = tmp[:, :-1], tmp[:, -1]
    y = y.astype(int)
    return X, y
