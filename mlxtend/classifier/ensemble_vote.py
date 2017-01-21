# Soft Voting/Majority Rule classifier

# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Implementation of an meta-classification algorithm for majority voting.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from ..externals.name_estimators import _name_estimators
from ..externals import six
import numpy as np


class EnsembleVoteClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """Soft Voting/Majority Rule classifier for scikit-learn estimators.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
        A list of classifiers.
        Invoking the `fit` method on the `VotingClassifier` will fit clones
        of those original classifiers that will
        be stored in the class attribute
        `self.clfs_` if `refit=True` (default).
    voting : str, {'hard', 'soft'} (default='hard')
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probalities, which is recommended for
        an ensemble of well-calibrated classifiers.
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurances of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the clf being fitted
        - `verbose=2`: Prints info about the parameters of the clf being fitted
        - `verbose>2`: Changes `verbose` param of the underlying clf to
           self.verbose - 2
    refit : bool (default: True)
        Refits classifiers in `clfs` if True; uses references to the `clfs`,
        otherwise (assumes that the classifiers were already fit).
        Note: refit=False is incompatible to mist scikit-learn wrappers!
        For instance, if any form of cross-validation is performed
        this would require the re-fitting classifiers to training folds, which
        would raise a NotFitterError if refit=False.
        (New in mlxtend v0.6.)

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]
    clf : array-like, shape = [n_predictions]
        The unmodified input classifiers
    clf_ : array-like, shape = [n_predictions]
        Fitted clones of the input classifiers

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleVoteClassifier
    >>> clf1 = LogisticRegression(random_seed=1)
    >>> clf2 = RandomForestClassifier(random_seed=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """
    def __init__(self, clfs, voting='hard',
                 weights=None, verbose=0, refit=True):

        self.clfs = clfs
        self.named_clfs = {key: value for key, value in _name_estimators(clfs)}
        self.voting = voting
        self.weights = weights
        self.verbose = verbose
        self.refit = refit

    def fit(self, X, y):
        """Learn weight coefficients from training data for each classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        if self.weights and len(self.weights) != len(self.clfs):
            raise ValueError('Number of classifiers and weights must be equal'
                             '; got %d weights, %d clfs'
                             % (len(self.weights), len(self.clfs)))

        self.le_ = LabelEncoder()
        self.le_.fit(y)
        self.classes_ = self.le_.classes_

        if not self.refit:
            self.clfs_ = [clf for clf in self.clfs]

        else:
            self.clfs_ = [clone(clf) for clf in self.clfs]

            if self.verbose > 0:
                print("Fitting %d classifiers..." % (len(self.clfs)))

            for clf in self.clfs_:

                if self.verbose > 0:
                    i = self.clfs_.index(clf) + 1
                    print("Fitting clf%d: %s (%d/%d)" %
                          (i, _name_estimators((clf,))[0][0], i,
                           len(self.clfs_)))

                if self.verbose > 2:
                    if hasattr(clf, 'verbose'):
                        clf.set_params(verbose=self.verbose - 2)

                if self.verbose > 1:
                    print(_name_estimators((clf,))[0][1])

                clf.fit(X, self.le_.transform(y))
        return self

    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.

        """
        if not hasattr(self, 'clfs_'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        if self.voting == 'soft':

            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            predictions = self._predict(X)

            maj = np.apply_along_axis(lambda x:
                                      np.argmax(np.bincount(x,
                                                weights=self.weights)),
                                      axis=1,
                                      arr=predictions)

        maj = self.le_.inverse_transform(maj)
        return maj

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.

        """
        if not hasattr(self, 'clfs_'):
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

        avg = np.average(self._predict_probas(X), axis=0, weights=self.weights)
        return avg

    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'` : array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'` : array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.

        """
        if self.voting == 'soft':
            return self._predict_probas(X)
        else:
            return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(EnsembleVoteClassifier, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict(X) for clf in self.clfs_]).T

    def _predict_probas(self, X):
        """Collect results from clf.predict_proba calls."""
        return np.asarray([clf.predict_proba(X) for clf in self.clfs_])
