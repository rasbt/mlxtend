# Stacking CV classifier

# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Authors: Reiichiro Nakano <github.com/reiinakano>
#          Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.name_estimators import _name_estimators
from ..externals.estimator_checks import check_is_fitted
import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.externals import six
from sklearn.model_selection._split import check_cv
from sklearn.utils import safe_indexing


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

    def fit(self, X, y, groups=None, sample_weight=None):
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

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list as well as the meta_regressor.
            Raises error if some regressor does not support
            sample_weight in the fit() method.

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

        folds = list(final_cv.split(X, y, groups))

        # Handle the case of X being a list of lists
        #   by converting X into a numpy array
        if isinstance(X, list):
            X = np.array(X)

        meta_features = None
        n_folds = final_cv.get_n_splits()
        n_models = len(self.clfs_)

        for n, model in enumerate(self.clfs_):

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

            for num, (train_indices, test_indices) in enumerate(folds):

                X_train = safe_indexing(X, train_indices)
                y_train = safe_indexing(y, train_indices)

                if self.verbose > 0:
                    print("Training and fitting fold %d of %d..." %
                          ((num + 1), n_folds))

                if sample_weight is None:
                    model.fit(X_train, y_train)
                else:
                    w = safe_indexing(sample_weight, train_indices)
                    model.fit(X_train, y_train, sample_weight=w)

                X_test = safe_indexing(X, test_indices)
                if not self.use_probas:
                    prediction = model.predict(X_test)[:, np.newaxis]
                else:
                    prediction = model.predict_proba(X_test)

                if meta_features is None:
                    # First run, use prediction to get the number of classes
                    n_classes = prediction.shape[1]
                    meta_features_shape = (X.shape[0], n_classes * n_models)
                    meta_features = np.empty(shape=meta_features_shape)
                    meta_features[np.array(test_indices)[:, np.newaxis],
                                  np.arange(n_classes)] = prediction
                else:
                    row_idx = np.array(test_indices)[:, np.newaxis]
                    col_idx = np.arange(n_classes) + n * n_classes
                    meta_features[row_idx, col_idx] = prediction

        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features

        # Fit the base models correctly this time using ALL the training set
        for model in self.clfs_:
            if sample_weight is None:
                model.fit(X, y)
            else:
                model.fit(X, y, sample_weight=sample_weight)

        # Fit the secondary model
        if self.use_features_in_secondary:
            meta_features = self._stack_first_level_features(
                X,
                meta_features
            )

        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y,
                               sample_weight=sample_weight)

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
        check_is_fitted(self, ['clfs_', 'meta_clf_'])

        per_model_preds = []

        for model in self.clfs_:
            if not self.use_probas:
                prediction = model.predict(X)[:, np.newaxis]
            else:
                prediction = model.predict_proba(X)

            per_model_preds.append(prediction)

        return np.hstack(per_model_preds)

    def _stack_first_level_features(self, X, meta_features):
        if sparse.issparse(X):
            stack_fn = sparse.hstack
        else:
            stack_fn = np.hstack

        return stack_fn((X, meta_features))

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X)

        if self.use_features_in_secondary:
            meta_features = self._stack_first_level_features(X, meta_features)

        return predict_fn(meta_features)

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
        check_is_fitted(self, ['clfs_', 'meta_clf_'])

        return self._do_predict(X, self.meta_clf_.predict)

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
        check_is_fitted(self, ['clfs_', 'meta_clf_'])

        return self._do_predict(X, self.meta_clf_.predict_proba)
