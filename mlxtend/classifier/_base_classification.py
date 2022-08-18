import numpy as np
from scipy import sparse
from sklearn.base import ClassifierMixin

from ..externals.estimator_checks import check_is_fitted


class _BaseStackingClassifier(ClassifierMixin):
    """Base class of stacking classifiers"""

    def _do_predict(self, X, predict_fn):
        meta_features = self.predict_meta_features(X)

        if not self.use_features_in_secondary:
            return predict_fn(meta_features)
        elif sparse.issparse(X):
            return predict_fn(sparse.hstack((X, meta_features)))
        else:
            return predict_fn(np.hstack((X, meta_features)))

    def predict(self, X):
        """Predict target values for X.

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
        check_is_fitted(self, ["clfs_", "meta_clf_"])

        return self._do_predict(X, self.meta_clf_.predict)

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
        check_is_fitted(self, ["clfs_", "meta_clf_"])

        return self._do_predict(X, self.meta_clf_.predict_proba)

    def decision_function(self, X):
        """ Predict class confidence scores for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        scores : shape=(n_samples,) if n_classes == 2 else \
            (n_samples, n_classes).
            Confidence scores per (sample, class) combination. In the binary
            case, confidence score for self.classes_[1] where >0 means this
            class would be predicted.

        """
        check_is_fitted(self, ["clfs_", "meta_clf_"])

        return self._do_predict(X, self.meta_clf_.decision_function)
