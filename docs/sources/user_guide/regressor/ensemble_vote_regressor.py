# Ensemble Voting Regressor

from ..externals.estimator_checks import check_is_fitted
from ..externals.name_estimators import _name_estimators
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import TransformerMixin
from sklearn.base import clone
from ..externals import six


class EnsembleVotingRegressor (BaseEstimator, RegressorMixin, TransformerMixin):

    """A Ensemble voting regressor for scikit-learn estimators for regression.

    Parameters
    ----------
    regressors : array-like, shape = [n_regressors]
        A list of regressors.
        Invoking the `fit` method on the `EnsembleVotingRegressor` will fit clones
        of those original regressors that will
        be stored in the class attribute
        `self.regr_`.
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
        Sequence of weights (`float` or `int`) to weight the occurances of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
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
    regressors : array-like, shape = [n_predictions]
        The unmodified input regressors
    regr_ : list, shape=[n_regressors]
        Fitted regressors (clones of the original regressors)
    refit : bool (default: True)
        Clones the regressors for stacking regression if True (default)
        or else uses the original ones, which will be refitted on the dataset
        upon calling the `fit` method. Setting refit=False is
        recommended if you are working with estimators that are supporting
        the scikit-learn fit/predict API interface but are not compatible
        to scikit-learn's `clone` function.

    """
    def __init__(self, regressors, weights=None, verbose=0, refit=True):

        self.regressors = regressors
        self.weights = weights
        self.verbose = verbose
        self.refit = refit
        self.named_clfs = {key: value for key, value in _name_estimators(regressors)}

    def fit(self, X, y, sample_weight=None):
        """Learn weight coefficients from training data for each classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Sample weights passed as sample_weights to each regressor
            in the regressors list .
            Raises error if some regressor does not support
            sample_weight in the fit() method.

        Returns
        -------
        self : object

        """
        if self.weights and len(self.weights) != len(self.regressors):
            raise ValueError('Number of regressors and weights must be equal'
                             '; got %d weights, %d regressors'
                             % (len(self.weights), len(self.regressors)))

        if not self.refit:
            self.regr_ = [clf for clf in self.regressors]

        else:
            self.regr_ = [clone(clf) for clf in self.regressors]

            if self.verbose > 0:
                print("Fitting %d regressors..." % (len(self.regressors)))

            for reg in self.regr_:

                if self.verbose > 0:
                    i = self.regr_.index(reg) + 1
                    print("Fitting clf%d: %s (%d/%d)" %
                          (i, _name_estimators((reg,))[0][0], i,
                           len(self.regr_)))

                if self.verbose > 2:
                    if hasattr(reg, 'verbose'):
                        reg.set_params(verbose=self.verbose - 2)

                if self.verbose > 1:
                    print(_name_estimators((reg,))[0][1])

                if sample_weight is None:
                    reg.fit(X, y)
                else:
                    reg.fit(X, y, sample_weight=sample_weight)
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
        check_is_fitted(self, 'regr_')
        res = np.average(self._predict(X), axis=1,
                          weights=self.weights)
        return res

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
        check_is_fitted(self, 'regr_')
        return self._predict(X)

    def get_params(self, deep=True):
        """Return estimator parameter names for GridSearch support."""
        if not deep:
            return super(EnsembleVotingRegressor, self).get_params(deep=False)
        else:
            out = self.named_clfs.copy()
            for name, step in six.iteritems(self.named_clfs):
                for key, value in six.iteritems(step.get_params(deep=True)):
                    out['%s__%s' % (name, key)] = value
            for key, value in six.iteritems(super(EnsembleVotingRegressor,
                                            self).get_params(deep=False)):
                out['%s' % key] = value
            return out

    def _predict(self, X):
        """Collect results from clf.predict calls."""
        return np.asarray([clf.predict(X) for clf in self.regr_]).T
