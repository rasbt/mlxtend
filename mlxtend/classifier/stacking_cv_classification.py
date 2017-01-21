# Stacking CV classifier

# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Authors: Reiichiro Nakano <github.com/reiinakano>
#          Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.name_estimators import _name_estimators
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.externals import six
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np


class StackingCVClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """A 'Stacking Cross-Validation' classifier for scikit-learn estimators.

    New in mlxtend v0.4.3

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
    n_folds : int (default=2)
        The number of folds used for while creating training data for the
        meta-classifier during fitting.
    use_features_in_secondary : bool (default: False)
        If True, the meta-classifier will be trained both on the predictions
        of the original classifiers and the original dataset.
        If False, the meta-classifier will be trained only on the predictions
        of the original classifiers.
    stratify : bool (default: True)
        If True, the cross-validation technique used for fitting the classifier
        will be Stratified K-Fold.
        If False, the cross-validation technique used will be Regular K-Fold.
        It is highly recommended to use Stratified K-Fold.
    shuffle : bool (default: True)
        If True, when fitting, the training data will be shuffled prior to
        cross-validation.
    random_state: None, int, or RandomState
        When shuffle=True, pseudo-random number generator state used for
        shuffling. If None, use default numpy RNG for shuffling.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
                       and which fold is currently being used for fitting
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
                 use_probas=False, n_folds=2,
                 use_features_in_secondary=False,
                 stratify=True, random_state=None,
                 shuffle=True, verbose=0):

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
        self.n_folds = n_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.stratify = stratify
        self.shuffle = shuffle
        self.random_state = random_state

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

        if self.stratify:
            skf = list(StratifiedKFold(n_splits=self.n_folds,
                                       shuffle=self.shuffle,
                                       random_state=self.random_state)
                       .split(X, y))
        else:
            skf = list(KFold(n_splits=self.n_folds,
                       shuffle=self.shuffle,
                       random_state=self.random_state).split(X))

        all_model_predictions = np.array([]).reshape(len(y), 0)
        for model in self.clfs_:

            if self.verbose > 0:
                i = self.clfs_.index(model) + 1
                print("Fitting classifier%d: %s (%d/%d)" %
                      (i, _name_estimators((model,))[0][0],
                       i, len(self.clfs_)))

            if self.verbose > 2:
                if hasattr(model, 'verbose'):
                    model.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((model,))[0][1])

            if not self.use_probas:
                single_model_prediction = np.array([]).reshape(0, 1)
            else:
                single_model_prediction = np.array([]).reshape(0, len(set(y)))

            for num, (train_index, test_index) in enumerate(skf):

                if self.verbose > 0:
                    print("Training and fitting fold %d of %d..." %
                          ((num + 1), self.n_folds))

                model.fit(X[train_index], y[train_index])

                if not self.use_probas:
                    prediction = model.predict(X[test_index])
                    prediction = prediction.reshape(prediction.shape[0], 1)
                else:
                    prediction = model.predict_proba(X[test_index])
                single_model_prediction = np.vstack([single_model_prediction.
                                                    astype(prediction.dtype),
                                                     prediction])

            all_model_predictions = np.hstack([all_model_predictions.
                                               astype(single_model_prediction.
                                                      dtype),
                                               single_model_prediction])

        # We have to shuffle the labels in the same order as we generated
        # predictions during CV (we kinda shuffled them when we did
        # Stratified CV).
        # We also do the same with the features (we will need this only IF
        # use_features_in_secondary is True)
        reordered_labels = np.array([]).astype(y.dtype)
        reordered_features = np.array([]).reshape((0, X.shape[1]))\
            .astype(X.dtype)
        for train_index, test_index in skf:
            reordered_labels = np.concatenate((reordered_labels,
                                               y[test_index]))
            reordered_features = np.concatenate((reordered_features,
                                                 X[test_index]))

        # Fit the base models correctly this time using ALL the training set
        for model in self.clfs_:
            model.fit(X, y)

        # Fit the secondary model
        if not self.use_features_in_secondary:
            self.meta_clf_.fit(all_model_predictions, reordered_labels)
        else:
            self.meta_clf_.fit(np.hstack((reordered_features,
                                          all_model_predictions)),
                               reordered_labels)

        return self

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(StackingCVClassifier, self).get_params(deep=False)
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
        all_model_predictions = np.array([]).reshape(len(X), 0)
        for model in self.clfs_:
            if not self.use_probas:
                single_model_prediction = model.predict(X)
                single_model_prediction = single_model_prediction\
                    .reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(X)
            all_model_predictions = np.hstack((all_model_predictions.
                                               astype(single_model_prediction
                                                      .dtype),
                                               single_model_prediction))
        if not self.use_features_in_secondary:
            return self.meta_clf_.predict(all_model_predictions)
        else:
            return self.meta_clf_.predict(np.hstack((X,
                                                     all_model_predictions)))

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
        all_model_predictions = np.array([]).reshape(len(X), 0)
        for model in self.clfs_:
            if not self.use_probas:
                single_model_prediction = model.predict(X)
                single_model_prediction = single_model_prediction\
                    .reshape(single_model_prediction.shape[0], 1)
            else:
                single_model_prediction = model.predict_proba(X)
            all_model_predictions = np.hstack((all_model_predictions.
                                               astype(single_model_prediction.
                                                      dtype),
                                               single_model_prediction))
        if not self.use_features_in_secondary:
            return self.meta_clf_.predict_proba(all_model_predictions)
        else:
            return self.meta_clf_\
                .predict_proba(np.hstack((X, all_model_predictions)))
