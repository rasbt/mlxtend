import numpy as np
from mlxtend.feature_selection import SFS


def test_Iris():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)

    sfs = SFS(knn, k_features=2, scoring='accuracy', cv=5)
    sfs.fit(X, y)

    assert(sfs.indices_ == (2, 3))
    assert(round(sfs.k_score_, 2) == 0.97 )
