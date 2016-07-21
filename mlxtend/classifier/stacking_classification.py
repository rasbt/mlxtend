# Stacking classifier

# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..externals import six
import numpy as np


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """A Stacking classifier for scikit-learn estimators for classification.

    Parameters
    ----------
    classifiers : array-like, shape = [n_regressors]
        A list of classifiers.
        Invoking the `fit` method on the `StackingClassifer` will fit clones
        of these original classifiers that will
        be stored in the class attribute
        `self.clfs_`.
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2

    Attributes
    ----------
    clfs_ : list, shape=[n_classifiers]
        Fitted classifiers (clones of the original classifiers)
    meta_clf_ : estimator
        Fitted meta-classifier (clone of the original meta-estimator)

    """
    def __init__(self, classifiers, meta_classifier,
                 use_probas=False, verbose=0):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.named_meta_classifier = {'meta-%s' % key: value for
                                      key, value in
                                      _name_estimators([meta_classifier])}
        self.use_probas = use_probas
        self.verbose = verbose

    def fit(self, X, y):
        """ Fit ensemble classifers and the meta-classifier.

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
        self.clfs_ = [clone(clf) for clf in self.classifiers]
        self.meta_clf_ = clone(self.meta_classifier)
        if self.verbose > 0:
            print("Fitting %d classifiers..." % (len(self.classifiers)))

        for clf in self.clfs_:

            if self.verbose > 0:
                i = self.clfs_.index(clf) + 1
                print("Fitting classifier%d: %s (%d/%d)" %
                      (i, _name_estimators((clf,))[0][0], i, len(self.clfs_)))

            if self.verbose > 2:
                if hasattr(clf, 'verbose'):
                    clf.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((clf,))[0][1])

            clf.fit(X, y)

        meta_features = self._predict_meta_features(X)
        self.meta_clf_.fit(meta_features, y)
        return self

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(StackingClassifier, self).get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in six.iteritems(self.named_classifiers):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_classifier.copy())
            for name, step in six.iteritems(self.named_meta_classifier):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            return out

    def _predict_meta_features(self, X):
        if self.use_probas:
            probas = np.asarray([clf.predict_proba(X) for clf in self.clfs_])
            vals = np.average(probas, axis=0)
        else:
            vals = np.asarray([clf.predict(X) for clf in self.clfs_]).T
        return vals

    def predict(self, X):
        """ Predict target values for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        labels : array-like, shape = [n_samples]
            Predicted class labels.

        """
        check_is_fitted(self, 'clfs_')
        meta_features = self._predict_meta_features(X)
        return self.meta_clf_.predict(meta_features)

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        proba : array-like, shape = [n_samples, n_classes]
            Probability for each class per sample.

        """
        check_is_fitted(self, 'clfs_')
        meta_features = self._predict_meta_features(X)
        return self.meta_clf_.predict_proba(meta_features)
