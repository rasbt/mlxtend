import numpy as np
from mlxtend.feature_selection import SBS


def test_Iris():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)

    sbs = SBS(knn, k_features=2, scoring='accuracy', cv=5, print_progress=False)
    sbs.fit(X, y)

    assert(sbs.indices_ == (0, 3))
    assert(round(sbs.k_score_, 2) == 0.96)


def test_selects_all():
    from sklearn.neighbors import KNeighborsClassifier
    from mlxtend.data import wine_data

    X, y = wine_data()
    knn = KNeighborsClassifier(n_neighbors=4)
    sbs = SBS(knn, k_features=13, scoring='accuracy', cv=3, print_progress=False)
    sbs.fit(X, y)
    assert(len(sbs.indices_) == 13)
