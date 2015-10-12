# Sebastian Raschka, 2015
# Function for sequential feature selection via
# Sequential Floating Backward Selection (SFBS)
# mlxtend Machine Learning Library Extensions

from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from itertools import combinations
from sklearn.cross_validation import cross_val_score
import numpy as np
from collections import deque
import sys

class SFBS(BaseEstimator, MetaEstimatorMixin):
    """ Sequential Floating Backward Selection for feature selection.

    Parameters
    ----------
    estimator : scikit-learn estimator object

    k_features : int
      Number of features to select where k_features.

    print_progress : bool (default: True)
       Prints progress as the number of epochs
       to stderr.

    scoring : str, (default='accuracy')
      Scoring metric for the cross validation scorer.

    cv : int (default: 5)
      Number of folds in StratifiedKFold.

    max_iter: int (default: -1)
      Terminate early if number of `max_iter` is reached.

    skip_if_stuck: bool (default: True)
      If `True`, skips conditional exclusion step if stuck.

    n_jobs : int (default: 1)
      The number of CPUs to use for cross validation. -1 means 'all CPUs'.

    Attributes
    ----------
    indices_ : array-like, shape = [n_predictions]
      Indices of the selected subsets.

    k_score_ : float
      Cross validation mean score of the selected subset

    subsets_ : list of lists
      Indices of the sequentially selected subsets.

    scores_ : list
      Cross validation mean scores of the sequentially selected subsets.

    Examples
    --------
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.datasets import load_iris
    >>> iris = load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> knn = KNeighborsClassifier(n_neighbors=4)
    >>> sfbs = SFBS(knn, k_features=2, scoring='accuracy', cv=5)
    >>> sfbs = sfbs.fit(X, y)
    >>> sfbs.indices_
    (2, 3)
    >>> sfbs.transform(X[:5])
    array([[ 1.4,  0.2],
           [ 1.4,  0.2],
           [ 1.3,  0.2],
           [ 1.5,  0.2],
           [ 1.4,  0.2]])

    >>> print('best score: %.2f' % sfbs.k_score_)
    best score: 0.97

    """
    def __init__(self, estimator, k_features, print_progress=True,
                 scoring='accuracy', max_iter=-1, cv=5,
                 skip_if_stuck=True, n_jobs=1):
        self.scoring = scoring
        self.estimator = estimator #clone(estimator)
        self.cv = cv
        self.max_iter = max_iter
        self.k_features = k_features
        self.skip_if_stuck = skip_if_stuck
        self.print_progress = print_progress
        self.n_jobs = n_jobs

    def fit(self, X, y):

        dim = X.shape[1]
        orig = tuple(range(dim))
        self.indices_ = orig
        self.subsets_ = [self.indices_]
        cv_score = self._calc_score(X, y, self.indices_)
        self.scores_ = [cv_score.mean()]

        if self.skip_if_stuck:
            sdq = deque(maxlen=4)
        else:
            sdq = deque(maxlen=0)

        cnt = 0
        while dim > self.k_features and cnt != self.max_iter:
            scores_1, scores_2 = [], []
            subsets_1, subsets_2 = [], []

            # step 1: exclusion
            for p in combinations(self.indices_, r=dim-1):
                cv_score = self._calc_score(X, y, p)
                scores_1.append(cv_score.mean())
                subsets_1.append(p)

            best_1 = np.argmax(scores_1)
            self.indices_ = subsets_1[best_1]
            excluded = set(orig) - set(self.indices_)

            # step 2: conditional inclusion
            best_2_score = -1.0
            if not len(sdq) == 4 or (sdq[0] != sdq[2] or sdq[1] != sdq[3]):
                set_indices = set(self.indices_)
                for i in excluded:
                    test_subset = tuple(sorted(set_indices | {i}))
                    cv_score = self._calc_score(X, y, test_subset)
                    scores_2.append(cv_score.mean())
                    subsets_2.append(test_subset)

                best_2 = np.argmax(scores_2)
                best_2_score = scores_2[best_2]


            if best_2_score > scores_1[best_1]:
                self.indices_ = subsets_2[best_2]
            else:
                self.subsets_.append(self.indices_)
                self.scores_.append(scores_1[best_1])
                dim -= 1

            sdq.append(self.indices_)
            cnt += 1

            if self.print_progress:
                sys.stderr.write('\rFeatures: %d/%d' % (len(self.indices_), self.k_features))
                sys.stderr.flush()

        self.k_score_ = self.scores_[-1]

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

    def _calc_score(self, X, y, indices):
        cv_score = cross_val_score(self.estimator,
                                   X[:, indices], y,
                                   cv=self.cv,
                                   scoring=self.scoring,
                                   n_jobs=self.n_jobs)
        return cv_score
