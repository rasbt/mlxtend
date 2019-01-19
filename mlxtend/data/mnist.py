# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source MNIST.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "mnist_5k.csv.gz")


def mnist_data():
    """5000 samples from the MNIST handwritten digits dataset.

    Data Source : http://yann.lecun.com/exdb/mnist/

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with 5000 image samples as rows,
        each row consists of 28x28 pixels that were unrolled into
        784 pixel feature vectors.
        y contains the 10 unique class labels 0-9.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/mnist_data/

    """
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
    X, y = tmp[:, :-1], tmp[:, -1]
    y = y.astype(int)
    return X, y
