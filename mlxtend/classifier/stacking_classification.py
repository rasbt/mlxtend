# Stacking classifier

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..externals import six
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
import numpy as np


class StackingClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """A Stacking classifier for scikit-learn estimators for classification.

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
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
    average_probas : bool (default: False)
        Averages the probabilities as meta features if True.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
    use_features_in_secondary : bool (default: False)
        If True, the meta-classifier will be trained both on the predictions
        of the original classifiers and the original dataset.
        If False, the meta-classifier will be trained only on the predictions
        of the original classifiers.
    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data used
        for fitting the meta-classifier stored in the
        `self.train_meta_features_` array, which can be
        accessed after calling `fit`.
    use_clones : bool (default: True)
        Clones the classifiers for stacking classification if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Hence, if use_clones=True, the original
        input classifiers will remain unmodified upon using the
        StackingClassifier's `fit` method.
        Setting `use_clones=False` is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    Attributes
    ----------
    clfs_ : list, shape=[n_classifiers]
        Fitted classifiers (clones of the original classifiers)
    meta_clf_ : estimator
        Fitted meta-classifier (clone of the original meta-estimator)
    train_meta_features : numpy array, shape = [n_samples, n_classifiers]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and n_classifiers is the number of classfiers.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/

    """
    def __init__(self, classifiers, meta_classifier,
                 use_probas=False, average_probas=False, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 use_clones=True):

        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.named_classifiers = {key: value for
                                  key, value in
                                  _name_estimators(classifiers)}
        self.named_meta_classifier = {'meta-%s' % key: value for
                                      key, value in
                                      _name_estimators([meta_classifier])}
        self.use_probas = use_probas
        self.average_probas = average_probas
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones

    def fit(self, X, y):
        """ Fit ensemble classifers and the meta-classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Target values.

        Returns
        -------
        self : object

        """
        if self.use_clones:
            self.clfs_ = [clone(clf) for clf in self.classifiers]
            self.meta_clf_ = clone(self.meta_classifier)
        else:
            self.clfs_ = self.classifiers
            self.meta_clf_ = self.meta_classifier

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

        meta_features = self.predict_meta_features(X)

        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        if not self.use_features_in_secondary:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(np.hstack((X, meta_features)), y)

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

            for key, value in six.iteritems(super(StackingClassifier,
                                            self).get_params(deep=False)):
                out['%s' % key] = value

            return out

    def predict_meta_features(self, X):
        """ Get meta-features of test-data.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        meta-features : numpy array, shape = [n_samples, n_classifiers]
            Returns the meta-features for test data.

        """
        check_is_fitted(self, 'clfs_')
        if self.use_probas:
            probas = np.asarray([clf.predict_proba(X)
                                 for clf in self.clfs_])
            if self.average_probas:
                vals = np.average(probas, axis=0)
            else:
                vals = np.concatenate(probas, axis=1)
        else:
            vals = np.column_stack([clf.predict(X) for clf in self.clfs_])
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
        labels : array-like, shape = [n_samples] or [n_samples, n_outputs]
            Predicted class labels.

        """
        check_is_fitted(self, 'clfs_')
        meta_features = self.predict_meta_features(X)

        if not self.use_features_in_secondary:
            return self.meta_clf_.predict(meta_features)
        else:
            return self.meta_clf_.predict(np.hstack((X, meta_features)))

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        proba : array-like, shape = [n_samples, n_classes] or a list of \
                n_outputs of such arrays if n_outputs > 1.
            Probability for each class per sample.

        """
        check_is_fitted(self, 'clfs_')
        meta_features = self.predict_meta_features(X)

        if not self.use_features_in_secondary:
            return self.meta_clf_.predict_proba(meta_features)
        else:
            return self.meta_clf_.predict_proba(np.hstack((X, meta_features)))
