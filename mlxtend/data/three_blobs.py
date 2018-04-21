# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# A function for loading a sample dataset for clustering evaluations
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "three_blobs.csv.gz")


def three_blobs_data():
    """A random dataset of 3 2D blobs for clustering.

    Number of samples : 150
    Suggested labels : {0, 1, 2}, distribution: [50, 50, 50]

    Returns
    --------
    X, y : [n_samples, n_features], [n_cluster_labels]
        X is the feature matrix with 159 samples as rows
        and 2 feature columns.
        y is a 1-dimensional array of the 3 suggested cluster labels 0, 1, 2

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/three_blobs_data

    """
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
    X, y = tmp[:, :-1], tmp[:, -1]
    y = y.astype(int)
    return X, y
