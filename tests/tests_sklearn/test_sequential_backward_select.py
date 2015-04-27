import numpy as np
from mlxtend.sklearn import SBS

def test_Iris():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)

    sbs = SBS(knn, k_features=2, scoring='accuracy', cv=5)
    sbs.fit(X, y)

    assert(sbs.indices_ == (0, 3))
    assert(sbs.k_score_ == 0.96)



