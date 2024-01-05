# Stacking CV classifier

# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-classifier for stacking
# Authors: Reiichiro Nakano <github.com/reiinakano>
#          Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import sparse
from sklearn.base import TransformerMixin, clone
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection._split import check_cv

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..utils.base_compostion import _BaseXComposition
from ._base_classification import _BaseStackingClassifier

# from sklearn.utils import check_X_y


class StackingCVClassifier(
    _BaseXComposition, _BaseStackingClassifier, TransformerMixin
):

    """A 'Stacking Cross-Validation' classifier for scikit-learn estimators.

    New in mlxtend v0.4.3

    Parameters
    ----------
    classifiers : array-like, shape = [n_classifiers]
        A list of classifiers.
        Invoking the `fit` method on the `StackingCVClassifer` will fit clones
        of these original classifiers that will
        be stored in the class attribute `self.clfs_` if `use_clones=True`.
    meta_classifier : object
        The meta-classifier to be fitted on the ensemble of
        classifiers
    use_probas : bool (default: False)
        If True, trains meta-classifier based on predicted probabilities
        instead of class labels.
    drop_proba_col : string (default: None)
        Drops extra "probability" column in the feature set, because it is
        redundant:
        p(y_c) = 1 - p(y_1) + p(y_2) + ... + p(y_{c-1}).
        This can be useful for meta-classifiers that are sensitive to perfectly
        collinear features.
        If 'last', drops last probability column.
        If 'first', drops first probability column.
        Only relevant if `use_probas=True`.
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
    shuffle : bool (default: True)
        If True,  and the `cv` argument is integer, the training data will be
        shuffled at fitting stage prior to cross-validation. If the `cv`
        argument is a specific cross validation technique, this argument is
        omitted.
    random_state : int, RandomState instance or None, optional (default: None)
        Constrols the randomness of the cv splitter. Used when `cv` is
        integer and `shuffle=True`. New in v0.16.0.
    stratify : bool (default: True)
        If True, and the `cv` argument is integer it will follow a stratified
        K-Fold cross validation technique. If the `cv` argument is a specific
        cross validation technique, this argument is omitted.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
                       and which fold is currently being used for fitting
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
        StackingCVClassifier's `fit` method.
        Setting `use_clones=False` is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.
    n_jobs : int or None, optional (default=None)
        The number of CPUs to use to do the computation.
        `None` means 1 unless in a `joblib.parallel_backend` context.
        `-1` means using all processors.
        for more details. New in v0.16.0.
    pre_dispatch : int, or string, optional
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:
            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs
            - An int, giving the exact number of total jobs that are
              spawned
            - A string, giving an expression as a function of n_jobs,
              as in '2*n_jobs'
        New in v0.16.0.

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
    https://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/

    """

    def __init__(
        self,
        classifiers,
        meta_classifier,
        use_probas=False,
        drop_proba_col=None,
        cv=2,
        shuffle=True,
        random_state=None,
        stratify=True,
        verbose=0,
        use_features_in_secondary=False,
        store_train_meta_features=False,
        use_clones=True,
        n_jobs=None,
        pre_dispatch="2*n_jobs",
    ):
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas

        allowed = {None, "first", "last"}
        if drop_proba_col not in allowed:
            raise ValueError(
                "`drop_proba_col` must be in %s. Got %s" % (allowed, drop_proba_col)
            )

        self.drop_proba_col = drop_proba_col
        self.cv = cv
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratify = stratify
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch

    @property
    def named_classifiers(self):
        return _name_estimators(self.classifiers)

    def fit(self, X, y, groups=None, sample_weight=None):
        """Fit ensemble classifers and the meta-classifier.

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
            self.clfs_ = clone(self.classifiers)
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
            final_cv.random_state = self.random_state

        # Disable global input validation, because it causes issue when
        # pipelines are used that perform preprocessing on X. I.e., X may
        # not be directly passed to the classifiers, which is why this code
        # would raise unecessary errors at this point.
        # X, y = check_X_y(X, y, accept_sparse=['csc', 'csr'], dtype=None)

        if sample_weight is None:
            fit_params = None
        else:
            fit_params = dict(sample_weight=sample_weight)

        meta_features = None

        for n, model in enumerate(self.clfs_):
            if self.verbose > 0:
                i = self.clfs_.index(model) + 1
                print(
                    "Fitting classifier%d: %s (%d/%d)"
                    % (i, _name_estimators((model,))[0][0], i, len(self.clfs_))
                )

            if self.verbose > 2:
                if hasattr(model, "verbose"):
                    model.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((model,))[0][1])

            prediction = cross_val_predict(
                model,
                X,
                y,
                groups=groups,
                cv=final_cv,
                n_jobs=self.n_jobs,
                fit_params=fit_params,
                verbose=self.verbose,
                pre_dispatch=self.pre_dispatch,
                method="predict_proba" if self.use_probas else "predict",
            )

            if not self.use_probas:
                prediction = prediction[:, np.newaxis]
            elif self.drop_proba_col == "last":
                prediction = prediction[:, :-1]
            elif self.drop_proba_col == "first":
                prediction = prediction[:, 1:]

            if meta_features is None:
                meta_features = prediction
            else:
                meta_features = np.column_stack((meta_features, prediction))

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
            meta_features = self._stack_first_level_features(X, meta_features)

        if sample_weight is None:
            self.meta_clf_.fit(meta_features, y)
        else:
            self.meta_clf_.fit(meta_features, y, sample_weight=sample_weight)

        return self

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        return self._get_params("named_classifiers", deep=deep)

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Valid parameter keys can be listed with ``get_params()``.

        Returns
        -------
        self
        """
        self._set_params("classifiers", "named_classifiers", **params)
        return self

    def predict_meta_features(self, X):
        """Get meta-features of test-data.

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
        check_is_fitted(self, ["clfs_", "meta_clf_"])

        per_model_preds = []

        for model in self.clfs_:
            if not self.use_probas:
                prediction = model.predict(X)[:, np.newaxis]
            else:
                if self.drop_proba_col == "last":
                    prediction = model.predict_proba(X)[:, :-1]
                elif self.drop_proba_col == "first":
                    prediction = model.predict_proba(X)[:, 1:]
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
