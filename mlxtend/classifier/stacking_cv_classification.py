# Stacking CV classifier

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Authors: Reiichiro Nakano <github.com/reiinakano>
#          Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.name_estimators import _name_estimators
from ..externals.estimator_checks import check_is_fitted
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals import six
from sklearn.model_selection._split import check_cv
import numpy as np


class StackingCVClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):

    """A 'Stacking Cross-Validation' classifier for scikit-learn estimators.

    New in mlxtend v0.4.3

    Notes
    -------
    The StackingCVClassifier uses scikit-learn's check_cv
    internally, which doesn't support a random seed. Thus
    NumPy's random seed need to be specified explicitely for
    deterministic behavior, for instance, by setting
    np.random.seed(RANDOM_SEED)
    prior to fitting the StackingCVClassifier

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        A list of classifiers.
        Invoking the `fit` method on the `StackingCVClassifer` will fit clones
        of these original classifiers that will
        be stored in the class attribute `self.clfs_`.
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    cv : int, cross-validation generator or an iterable, optional (default: 2)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 2-fold cross validation,
          - integer, to specify the number of folds in a `(Stratified)KFold`,
          - An object to be used as a cross-validation generator.
          - An iterable yielding train, test splits.
        For integer/None inputs, it will use either a `KFold` or
        `StratifiedKFold` cross validation depending the value of `stratify`
        argument.
    use_features_in_secondary : bool (default: False)
        If True, the meta-classifier will be trained both on the predictions
        of the original classifiers and the original dataset.
        If False, the meta-classifier will be trained only on the predictions
        of the original classifiers.
    stratify : bool (default: True)
        If True, and the `cv` argument is integer it will follow a stratified
        K-Fold cross validation technique. If the `cv` argument is a specific
        cross validation technique, this argument is omitted.
    shuffle : bool (default: True)
        If True,  and the `cv` argument is integer, the training data will be
        shuffled at fitting stage prior to cross-validation. If the `cv`
        argument is a specific cross validation technique, this argument is
        omitted.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
                       and which fold is currently being used for fitting
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
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
        StackingCVClassifier's `fit` method.
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
    http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/

    """
    def __init__(self, classifiers, meta_classifier,
                 use_probas=False, cv=2,
                 use_features_in_secondary=False,
                 stratify=True,
                 shuffle=True, verbose=0,
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
        self.verbose = verbose
        self.cv = cv
        self.use_features_in_secondary = use_features_in_secondary
        self.stratify = stratify
        self.shuffle = shuffle
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones

    def fit(self, X, y, groups=None):
        """ Fit ensemble classifers and the meta-classifier.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : numpy array, shape = [n_samples]
            Target values.

        groups : numpy array/None, shape = [n_samples]
            The group that each sample belongs to. This is used by specific
            folding strategies such as GroupKFold()

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

        final_cv = check_cv(self.cv, y, classifier=self.stratify)
        if isinstance(self.cv, int):
            # Override shuffle parameter in case of self generated
            # cross-validation strategy
            final_cv.shuffle = self.shuffle
        skf = list(final_cv.split(X, y, groups))

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
                          ((num + 1), final_cv.get_n_splits()))

                try:
                    model.fit(X[train_index], y[train_index])
                except TypeError as e:
                    raise TypeError(str(e) + '\nPlease check that X and y'
                                    'are NumPy arrays. If X and y are lists'
                                    ' of lists,\ntry passing them as'
                                    ' numpy.array(X)'
                                    ' and numpy.array(y).')
                except KeyError as e:
                    raise KeyError(str(e) + '\nPlease check that X and y'
                                   ' are NumPy arrays. If X and y are pandas'
                                   ' DataFrames,\ntry passing them as'
                                   ' X.values'
                                   ' and y.values.')

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

        if self.store_train_meta_features:
            # Store the meta features in the order of the
            # original X,y arrays
            reodered_indices = np.array([]).astype(y.dtype)
            for train_index, test_index in skf:
                reodered_indices = np.concatenate((reodered_indices,
                                                   test_index))
            self.train_meta_features_ = all_model_predictions[np.argsort(
                reodered_indices)]

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

            for key, value in six.iteritems(super(StackingCVClassifier,
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
        return all_model_predictions

    def predict(self, X):
        """ Predict target values for X.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        labels : array-like, shape = [n_samples]
            Predicted class labels.

        """
        check_is_fitted(self, 'clfs_')
        all_model_predictions = self.predict_meta_features(X)
        if not self.use_features_in_secondary:
            return self.meta_clf_.predict(all_model_predictions)
        else:
            return self.meta_clf_.predict(np.hstack((X,
                                                     all_model_predictions)))

    def predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
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
