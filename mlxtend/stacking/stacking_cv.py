# Stacking CV estimators

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for stacking regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv

from ..externals.estimator_checks import check_is_fitted
from .stacking import StackingEstimator
from .stacking import StackingRegressor
from .stacking import StackingClassifier


class StackingCVEstimator(StackingEstimator):
    def __init__(self, estimators, meta_estimator,
                 cv=5, shuffle=True, stratify=False, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 use_clones=True):
        super(StackingCVEstimator, self).__init__(
            estimators, meta_estimator, verbose=verbose,
            use_features_in_secondary=use_features_in_secondary,
            store_train_meta_features=store_train_meta_features,
            use_clones=use_clones)
        self.cv = cv
        self.shuffle = shuffle
        self.stratify = stratify

    def fit(self, X, y, groups=None, sample_weight=None):
        """ Fit ensemble estimator and the meta-estimator.

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
        # initialize estimators
        # make copy or assign references
        self._initialize_estimators()
        
        meta_features = self._meta_features_cv(
            X, y, groups=groups, sample_weight=sample_weight)
        # save meta-features for training data
        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features
                      
        # fit base estimators with full training samples,
        # which used at prediction phase
        self._fit_base_estimators(X, y, sample_weight=sample_weight)

        # add variables for meta regression, if needed
        meta_features = self._augment_meta_features(X, meta_features)

        self._fit_one(self.meta_est_, meta_features, y, 
                      sample_weight=sample_weight)

        return self
        
    def _meta_features_cv(self, X, y, 
                          groups=None, sample_weight=None):
        # In stacking CV models, meta features are predicted by
        # temporarily fitting to partial traing samples and
        # applying to the holdout samples.
        # Hence requires arguments for fitting.
        
        kfold = check_cv(self.cv, y, classifier=self.stratify)
        if isinstance(self.cv, int):
            # Override shuffle parameter in case of self generated
            # cross-validation strategy
            kfold.shuffle = self.shuffle

        meta_features = None
        #
        # The outer loop iterates over the base-regressors. Each regressor
        # is trained cv times and makes predictions, after which we train
        # the meta-regressor on their combined results.
        #
        for i, estimator in enumerate(self.estimators_):
            #
            # In the inner loop, each model is trained cv times on the
            # training-part of this fold of data; and the holdout-part of data
            # is used for predictions. This is repeated cv times, so in
            # the end we have predictions for each data point.
            #
            # Advantage of this complex approach is that data points we're
            # predicting have not been trained on by the algorithm, so it's
            # less susceptible to overfitting.
            #
            this_meta_features = None
            for train_idx, holdout_idx in kfold.split(X, y, groups):
                instance = clone(estimator)
                self._fit_one(
                    instance, X[train_idx], y[train_idx],
                    sample_weight=None if sample_weight is None \
                                       else sample_weight[train_idx])

                y_pred = self._predict_meta_feature_one(
                    instance, X[holdout_idx])

                # make sure prediction is two dimensional for
                # accumulating consistency
                assert len(y_pred.shape) < 3, "y must be 2d or smaller"
                if len(y_pred.shape) == 1:
                    y_pred = np.expand_dims(y_pred, axis=1)
                # initialize output here by looking at the shape of 
                # prediction outcome, since guessing is not so easy :p
                if this_meta_features is None:
                    this_meta_features = np.zeros((len(y), y_pred.shape[1]))
                this_meta_features[holdout_idx] = y_pred
            if meta_features is None:
                meta_features = this_meta_features
            else:
                meta_features = np.concatenate(
                    [meta_features, this_meta_features], axis=1)
        return meta_features

    def _predict_meta_feature_one(self, model, X):
        return model.predict(X)
        

class StackingCVRegressor(StackingCVEstimator, StackingRegressor):
    """A 'Stacking Cross-Validation' regressor for scikit-learn estimators.

    New in mlxtend v0.7.0

    Notes
    -------
    The StackingCVRegressor uses scikit-learn's check_cv
    internally, which doesn't support a random seed. Thus
    NumPy's random seed need to be specified explicitely for
    deterministic behavior, for instance, by setting
    np.random.seed(RANDOM_SEED)
    prior to fitting the StackingCVRegressor

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `StackingCVRegressor` will fit clones
        of these original regressors that will
        be stored in the class attribute `self.regr_`.
    meta_regressor : object
        The meta-regressor to be fitted on the ensemble of
        regressor
    cv : int, cross-validation generator or iterable, optional (default: 5)
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a `KFold`,
        - An object to be used as a cross-validation generator.
        - An iterable yielding train, test splits.
        For integer/None inputs, it will use `KFold` cross-validation
    shuffle : bool (default: True)
        If True,  and the `cv` argument is integer, the training data will
        be shuffled at fitting stage prior to cross-validation. If the `cv`
        argument is a specific cross validation technique, this argument is
        omitted.
    verbose : int, optional (default=0)
        Controls the verbosity of the building process.
        - `verbose=0` (default): Prints nothing
        - `verbose=1`: Prints the number & name of the regressor being fitted
        - `verbose=2`: Prints info about the parameters of the
                       regressor being fitted
        - `verbose>2`: Changes `verbose` param of the underlying regressor to
           self.verbose - 2
    use_features_in_secondary : bool (default: False)
        If True, the meta-regressor will be trained both on
        the predictions of the original regressors and the
        original dataset.
        If False, the meta-regressor will be trained only on
        the predictions of the original regressors.
    store_train_meta_features : bool (default: False)
        If True, the meta-features computed from the training data
        used for fitting the
        meta-regressor stored in the `self.train_meta_features_` array,
        which can be
        accessed after calling `fit`.
    use_clones : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    Attributes
    ----------
    train_meta_features : numpy array, shape = [n_samples, n_regressors]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and len(self.regressors) is the number of regressors.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/

    """
    def __init__(self, regressors, meta_regressor,
                 cv=5, shuffle=True, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False,
                 use_clones=True):
        super(StackingCVRegressor, self).__init__(
            regressors, meta_regressor,
            cv=cv, shuffle=shuffle, stratify=False, verbose=verbose,
            use_features_in_secondary=use_features_in_secondary,
            store_train_meta_features=store_train_meta_features,
            use_clones=use_clones)
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        

class StackingCVClassifier(StackingCVEstimator, StackingClassifier):

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
        super(StackingCVClassifier, self).__init__(
            classifiers, meta_classifier,
            cv=cv, shuffle=shuffle, stratify=stratify, verbose=verbose,
            use_features_in_secondary=use_features_in_secondary,
            store_train_meta_features=store_train_meta_features,
            use_clones=use_clones)
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        self.use_probas = use_probas
    
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
                                                            
    def _predict_meta_feature_one(self, model, X):
        if not self.use_probas:
            prediction = model.predict(X)
            prediction = prediction.reshape(prediction.shape[0], 1)
        else:
            prediction = model.predict_proba(X)
        if len(prediction.shape) > 0:
            prediction = np.squeeze(prediction)
        return prediction
