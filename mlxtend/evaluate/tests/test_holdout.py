# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.evaluate import RandomHoldoutSplit
from mlxtend.data import iris_data


X, y = iris_data()


def test_default_iter():
    h_iter = RandomHoldoutSplit(valid_size=0.3, random_seed=123)

    cnt = 0
    for train_ind, valid_ind in h_iter.split(X, y):
        cnt += 1

    assert cnt == 1

    expect_train_ind = np.array([60, 16, 88, 130, 6, 149, 7, 139, 83, 4,
                                 147, 111, 108, 11, 44, 36, 31, 35, 40, 119,
                                 107, 24, 103, 87, 10, 82, 29, 106, 104,
                                 90, 118, 132, 129, 84, 85, 43, 74, 94,
                                 73, 62, 69, 133, 52, 23, 1, 134, 96, 14, 50,
                                 128, 8, 140, 0, 18, 55, 67, 75, 120, 78,
                                 141, 126, 15, 5, 109, 102, 114, 3, 142, 26,
                                 79, 71, 136, 137, 21, 48, 47, 70, 92,
                                 124, 53, 121, 131, 54, 59, 81, 98, 49, 30,
                                 99, 51, 112, 113, 27, 46, 97, 123, 89, 145,
                                 95, 13, 58, 41, 12, 20, 148])
    expect_valid_ind = np.array([72, 125, 80, 86, 117, 17, 33, 68, 2, 28, 122,
                                 19, 116, 22, 91, 9, 146, 101, 38, 39, 37, 42,
                                 25, 65, 32, 56, 115, 34, 76, 143, 144, 100,
                                 138, 57, 93, 61, 127, 77, 110, 45, 135,
                                 64, 66, 105, 63])
    assert (train_ind == expect_train_ind).all()
    assert (valid_ind == expect_valid_ind).all()


def test_in_sfs():
    h_iter = RandomHoldoutSplit(valid_size=0.3, random_seed=123)
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(knn,
               k_features=3,
               forward=True,
               floating=False,
               verbose=2,
               scoring='accuracy',
               cv=h_iter)

    sfs1 = sfs1.fit(X, y)
    d = sfs1.get_metric_dict()
    assert d[1]['cv_scores'].shape[0] == 1


def test_in_grid():
    params = {'n_neighbors': [1, 2, 3, 4, 5]}

    grid = GridSearchCV(KNeighborsClassifier(),
                        param_grid=params,
                        cv=RandomHoldoutSplit(valid_size=0.3, random_seed=123))
    grid.fit(X, y)
    assert grid.n_splits_ == 1
