from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from collections import defaultdict
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=4)


sets = defaultdict(list)
for i in range(100):
    sfs1 = SFS(knn,
               k_features=4,
               forward=True,
               floating=False,
               verbose=0,
               scoring='accuracy',
               fuzzy=True,
               threshold=.01,
               cv=3,
               random_seed=i)

    sfs1 = sfs1.fit(X, y)
    sets[tuple(sfs1.k_feature_idx_)].append(sfs1.k_score_)
sets = {k: (np.mean(v), np.std(v)) for k, v in sets.items()}
print sets
