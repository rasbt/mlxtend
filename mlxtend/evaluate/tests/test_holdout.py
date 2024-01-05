# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from packaging.version import Version
from sklearn import __version__ as sklearn_version
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from mlxtend.data import iris_data
from mlxtend.evaluate import PredefinedHoldoutSplit, RandomHoldoutSplit
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

X, y = iris_data()


def test_randomholdoutsplit_default_iter():
    h_iter = RandomHoldoutSplit(valid_size=0.3, random_seed=123)

    cnt = 0
    for train_ind, valid_ind in h_iter.split(X, y):
        cnt += 1

    assert cnt == 1

    expect_train_ind = np.array(
        [
            60,
            16,
            88,
            130,
            6,
            149,
            7,
            139,
            83,
            4,
            147,
            111,
            108,
            11,
            44,
            36,
            31,
            35,
            40,
            119,
            107,
            24,
            103,
            87,
            10,
            82,
            29,
            106,
            104,
            90,
            118,
            132,
            129,
            84,
            85,
            43,
            74,
            94,
            73,
            62,
            69,
            133,
            52,
            23,
            1,
            134,
            96,
            14,
            50,
            128,
            8,
            140,
            0,
            18,
            55,
            67,
            75,
            120,
            78,
            141,
            126,
            15,
            5,
            109,
            102,
            114,
            3,
            142,
            26,
            79,
            71,
            136,
            137,
            21,
            48,
            47,
            70,
            92,
            124,
            53,
            121,
            131,
            54,
            59,
            81,
            98,
            49,
            30,
            99,
            51,
            112,
            113,
            27,
            46,
            97,
            123,
            89,
            145,
            95,
            13,
            58,
            41,
            12,
            20,
            148,
        ]
    )
    expect_valid_ind = np.array(
        [
            72,
            125,
            80,
            86,
            117,
            17,
            33,
            68,
            2,
            28,
            122,
            19,
            116,
            22,
            91,
            9,
            146,
            101,
            38,
            39,
            37,
            42,
            25,
            65,
            32,
            56,
            115,
            34,
            76,
            143,
            144,
            100,
            138,
            57,
            93,
            61,
            127,
            77,
            110,
            45,
            135,
            64,
            66,
            105,
            63,
        ]
    )
    assert (train_ind == expect_train_ind).all()
    assert (valid_ind == expect_valid_ind).all()


def test_randomholdoutsplit_in_sfs():
    h_iter = RandomHoldoutSplit(valid_size=0.3, random_seed=123)
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        verbose=2,
        scoring="accuracy",
        cv=h_iter,
    )

    sfs1 = sfs1.fit(X, y)
    d = sfs1.get_metric_dict()
    assert d[1]["cv_scores"].shape[0] == 1


def test_randomholdoutsplit_in_grid():
    params = {"n_neighbors": [1, 2, 3, 4, 5]}

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            KNeighborsClassifier(),
            iid=False,
            param_grid=params,
            cv=RandomHoldoutSplit(valid_size=0.3, random_seed=123),
        )
    else:
        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=params,
            cv=RandomHoldoutSplit(valid_size=0.3, random_seed=123),
        )

    grid.fit(X, y)
    assert grid.n_splits_ == 1


def test_predefinedholdoutsplit_default_iter():
    h_iter = PredefinedHoldoutSplit(valid_indices=[0, 1, 99])

    cnt = 0
    for train_ind, valid_ind in h_iter.split(X, y):
        cnt += 1

    assert cnt == 1

    expect_train_ind = np.array(
        [
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
            68,
            69,
            70,
            71,
            72,
            73,
            74,
            75,
            76,
            77,
            78,
            79,
            80,
            81,
            82,
            83,
            84,
            85,
            86,
            87,
            88,
            89,
            90,
            91,
            92,
            93,
            94,
            95,
            96,
            97,
            98,
            100,
            101,
            102,
            103,
            104,
            105,
            106,
            107,
            108,
            109,
            110,
            111,
            112,
            113,
            114,
            115,
            116,
            117,
            118,
            119,
            120,
            121,
            122,
            123,
            124,
            125,
            126,
            127,
            128,
            129,
            130,
            131,
            132,
            133,
            134,
            135,
            136,
            137,
            138,
            139,
            140,
            141,
            142,
            143,
            144,
            145,
            146,
            147,
            148,
            149,
        ]
    )
    expect_valid_ind = np.array([0, 1, 99])
    np.testing.assert_equal(train_ind, expect_train_ind)
    assert (valid_ind == expect_valid_ind).all()


def test_predefinedholdoutsplit_in_sfs():
    h_iter = PredefinedHoldoutSplit(valid_indices=[0, 1, 99])
    knn = KNeighborsClassifier(n_neighbors=4)

    sfs1 = SFS(
        knn,
        k_features=3,
        forward=True,
        floating=False,
        verbose=2,
        scoring="accuracy",
        cv=h_iter,
    )

    sfs1 = sfs1.fit(X, y)
    d = sfs1.get_metric_dict()
    assert d[1]["cv_scores"].shape[0] == 1


def test_predefinedholdoutsplit_in_grid():
    params = {"n_neighbors": [1, 3, 5]}

    if Version(sklearn_version) < Version("0.24.1"):
        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=params,
            iid=False,
            cv=PredefinedHoldoutSplit(valid_indices=[0, 1, 99]),
        )
    else:
        grid = GridSearchCV(
            KNeighborsClassifier(),
            param_grid=params,
            cv=PredefinedHoldoutSplit(valid_indices=[0, 1, 99]),
        )
    grid.fit(X, y)
    assert grid.n_splits_ == 1
