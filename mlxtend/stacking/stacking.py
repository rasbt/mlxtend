# Stacking estimators

# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# An ensemble-learning meta-regressor for stacking regression
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from ..externals import six
import numpy as np
import scipy.sparse as sparse
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from sklearn.model_selection import check_cv


class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimators, meta_estimator, verbose=0,
                 use_features_in_secondary=False, 
                 store_train_meta_features=True, use_clones=True):
        self.estimators = estimators
        self.meta_estimator = meta_estimator
        self.verbose = verbose
        self.use_features_in_secondary = use_features_in_secondary
        self.store_train_meta_features = store_train_meta_features
        self.use_clones = use_clones
        
        # Placeholders for ests_, meta_est_
        #self.ests_, self.meta_est_ = estimators, meta_estimator

    def fit(self, X, y, sample_weight=None):
        """Learn weight coefficients from training data for each regressor.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples] or [n_samples, n_targets]
            Target values.
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
        
        # fit base estimators
        self._fit_base_estimators(X, y, sample_weight=sample_weight)

        
        meta_features = self.predict_meta_features(X)
        # save meta-features for training data
        if self.store_train_meta_features:
            self.train_meta_features_ = meta_features
        
        # add variables for meta regression, if needed
        meta_features = self._augment_meta_features(X, meta_features)

        self._fit_one(self.meta_est_, meta_features, y, 
                      sample_weight=sample_weight)
 
        return self

    def predict(self, X):
        """ Predict target values for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        y_target : array-like, shape = [n_samples] or [n_samples, n_targets]
            Predicted target values.
        """
        check_is_fitted(self, 'ests_')
        check_is_fitted(self, 'meta_est_')

        meta_features = self.predict_meta_features(X)
        meta_features = self._augment_meta_features(X, meta_features)
        return self.meta_est_.predict(meta_features)
                    
    def predict_meta_features(self, X):
        """ Get meta-features of test-data.

        Parameters
        ----------
        X : numpy array, shape = [n_samples, n_features]
            Test vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        meta-features : numpy array, shape = [n_samples, len(self.regressors)]
            meta-features for test data, where n_samples is the number of
            samples in test data and len(self.regressors) is the number
            of regressors.

        """
        check_is_fitted(self, 'ests_')
        return np.column_stack([r.predict(X) for r in self.ests_])

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(StackingEstimator, self).get_params(deep=False)
        else:
            out = self.named_estimators.copy()
            for name, step in six.iteritems(self.named_estimators):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            out.update(self.named_meta_estimator.copy())
            for name, step in six.iteritems(self.named_meta_estimator):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value

            for key, value in six.iteritems(super(StackingEstimator,
                                            self).get_params(deep=False)):
                out['%s' % key] = value

            return out

    def _initialize_estimators(self):
        # if use_clones, create copies of base estimators
        # otherwise we assign the references
        if self.use_clones:
            self.ests_ = [clone(e) for e in self.estimators_]
            self.meta_est_ = clone(self.meta_estimator_)
        else:
            self.ests_ = self.estimators_
            self.meta_est_ = self.meta_estimator_
    
    def _fit_one(self, estimator, X, y, sample_weight=None):
        if sample_weight is None:
            estimator.fit(X, y)
        else:
            estimator.fit(X, y, sample_weight=sample_weight)
    
    def _fit_base_estimators(self, X, y, sample_weight=None):
        if self.verbose > 0:
            print("Fitting %d regressors..." % len(self.ests_))

        for estimator in self.ests_:

            if self.verbose > 0:
                i = self.ests_.index(estimator) + 1
                print("Fitting regressor%d: %s (%d/%d)" %
                      (i, _name_estimators((estimator,))[0][0],
                       i, len(self.ests_)))

            if self.verbose > 2:
                if hasattr(estimator, 'verbose'):
                    estimator.set_params(verbose=self.verbose - 2)

            if self.verbose > 1:
                print(_name_estimators((estimator,))[0][1])
            
            self._fit_one(estimator, X, y, sample_weight=sample_weight)

    def _augment_meta_features(self, X, prediction_features):
        if not self.use_features_in_secondary:
            # meta model uses the prediction outcomes only
            return prediction_features
        elif sparse.issparse(X):
            return sparse.hstack((X, prediction_features))
        else:
            return np.hstack((X, prediction_features))

    @property
    def estimators_(self):
        return self.estimators

    @property
    def meta_estimator_(self):
        return self.meta_estimator

    @property    
    def named_estimators(self):
        return {key: value \
                for key, value in _name_estimators(self.estimators_)}
    
    @property
    def named_meta_estimator(self):
        return {'meta-%s' % key: value \
                for key, value in _name_estimators([self.meta_estimator_])}
        
class StackingRegressor(StackingEstimator, RegressorMixin):

    """A Stacking regressor for scikit-learn estimators for regression.

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `StackingRegressor` will fit clones
        of those original regressors that will
        be stored in the class attribute
        `self.regr_`.
    meta_regressor : object
        The meta-regressor to be fitted on the ensemble of
        regressors
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


    Attributes
    ----------
    regr_ : list, shape=[n_regressors]
        Fitted regressors (clones of the original regressors)
    meta_regr_ : estimator
        Fitted meta-regressor (clone of the original meta-estimator)
    coef_ : array-like, shape = [n_features]
        Model coefficients of the fitted meta-estimator
    intercept_ : float
        Intercept of the fitted meta-estimator
    train_meta_features : numpy array, shape = [n_samples, len(self.regressors)]
        meta-features for training data, where n_samples is the
        number of samples
        in training data and len(self.regressors) is the number of regressors.
    use_clones : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.
        
    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/

    """
    def __init__(self, regressors, meta_regressor, verbose=0,
                 use_features_in_secondary=False,
                 store_train_meta_features=False, use_clones=True):
        super(StackingRegressor, self).__init__(
            regressors, meta_regressor, verbose=verbose,
            use_features_in_secondary=use_features_in_secondary,
            store_train_meta_features=store_train_meta_features,
            use_clones=use_clones)
        self.regressors = regressors
        self.meta_regressor = meta_regressor
        del self.estimators
        del self.meta_estimator
        
    @property
    def estimators_(self):
        return self.regressors

    @property
    def meta_estimator_(self):
        return self.meta_regressor

    @property
    def regr_(self):
        return self.ests_
        
    @property
    def meta_regr_(self):
        return self.meta_est_

    @property
    def intercept_(self):
        return self.meta_est_.intercept_
    
    @property
    def coef_(self):
        return self.meta_est_.coef_


class StackingClassifier(StackingEstimator, ClassifierMixin):

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
        super(StackingClassifier, self).__init__(
            classifiers, meta_classifier, verbose=verbose,
            use_features_in_secondary=use_features_in_secondary,
            store_train_meta_features=store_train_meta_features,
            use_clones=use_clones)
        self.use_probas = use_probas
        self.average_probas = average_probas    
        self.classifiers = classifiers
        self.meta_classifier = meta_classifier
        del self.estimators
        del self.meta_estimator

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
        check_is_fitted(self, 'meta_clf_')
        meta_features = self.predict_meta_features(X)
        meta_features = self._augment_meta_features(X, meta_features)
        return self.meta_clf_.predict_proba(meta_features)

    @property
    def estimators_(self):
        return self.classifiers

    @property
    def meta_estimator_(self):
        return self.meta_classifier
        
    @property
    def clfs_(self):
        return self.ests_

    @property
    def meta_clf_(self):
        return self.meta_est_