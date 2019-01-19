# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for loading the open-source AutoMPG dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import os

this_dir, this_filename = os.path.split(__file__)
DATA_PATH = os.path.join(this_dir, "data", "autompg.csv.gz")


def autompg_data():
    """Auto MPG dataset.


    Source : https://archive.ics.uci.edu/ml/datasets/Auto+MPG
    Number of samples : 392
    Continuous target variable : mpg

    Dataset Attributes:

        - 1) cylinders:  multi-valued discrete
        - 2) displacement: continuous
        - 3) horsepower: continuous
        - 4) weight: continuous
        - 5) acceleration: continuous
        - 6) model year: multi-valued discrete
        - 7) origin: multi-valued discrete
        - 8) car name: string (unique for each instance)

    Returns
    --------
    X, y : [n_samples, n_features], [n_targets]
        X is the feature matrix with 392 auto samples as rows
        and 8 feature columns (6 rows with NaNs removed).
        y is a 1-dimensional array of the target MPG values.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/autompg_data/

    """
    tmp = np.genfromtxt(fname=DATA_PATH, delimiter=',')
    X, y = tmp[:, :-1], tmp[:, -1]
    return X, y
