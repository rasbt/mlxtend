# OneR classifier

# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# The classic OneR (One Rule) classifier
# Authors: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings

import numpy as np
from scipy.stats import chi2_contingency
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError


class OneRClassifier(BaseEstimator, ClassifierMixin):
    """OneR (One Rule) Classifier.

    Parameters
    ----------
    resolve_ties : str (default: 'first')
        Option for how to resolve ties if two or more features
        have the same error. Options are
        - 'first' (default): chooses first feature in the list, i.e.,
          feature with the lower column index.
        - 'chi-squared': performs a chi-squared test for each feature
          against the target and selects the feature with the lowest p-value.

    Attributes
    ----------
    self.classes_labels_ : array-like, shape = [n_labels]
        Array containing the unique class labels found in the
        training set.

    self.feature_idx_ : int
        The index of the rules' feature based on the column in
        the training set.

    self.p_value_ : float
        The p value for a given feature. Only available after calling `fit`
        when the OneR attribute `resolve_ties = 'chi-squared'` is set.

    self.prediction_dict_ : dict
        Dictionary containing information about the
        feature's (self.feature_idx_)
        rules and total error. E.g.,
        `{'total error': 37, 'rules (value: class)': {0: 0, 1: 2}}`
        means the total error is 37, and the rules are
        "if feature value == 0 classify as 0"
        and "if feature value == 1 classify as 2".
        (And classify as class 1 otherwise.)

    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/classifier/OneRClassifier/
    """

    def __init__(self, resolve_ties="first"):
        allowed = {"first", "chi-squared"}
        if resolve_ties not in allowed:
            raise ValueError(
                "resolve_ties must be in %s. Got %s." % (allowed, resolve_ties)
            )
        self.resolve_ties = resolve_ties

    def fit(self, X, y):
        """Learn rule from training data.

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
        # This check will only catch the most extreme cases
        # but better than nothing
        for c in range(X.shape[1]):
            if np.unique(X[:, c]).shape[0] == X.shape[0]:
                warnings.warn(
                    "Feature array likely contains at least one"
                    " non-categorical column."
                    " Column %d appears to have a unique value"
                    " in every row." % c
                )
            break

        n_class_labels = np.unique(y).shape[0]

        def compute_class_counts(X, y, feature_index, feature_value):
            mask = X[:, feature_index] == feature_value
            return np.bincount(y[mask], minlength=n_class_labels)

        prediction_dict = {}  # save feature_idx: feature_val, label, error

        # iterate over features
        for feature_index in np.arange(X.shape[1]):
            # iterate over each possible value per feature
            for feature_value in np.unique(X[:, feature_index]):
                class_counts = compute_class_counts(X, y, feature_index, feature_value)
                most_frequent_class = np.argmax(class_counts)
                self.class_labels_ = np.unique(y)

                # count all classes for that feature match
                # except the most frequent one
                inverse_index = np.ones(n_class_labels, dtype=bool)
                inverse_index[most_frequent_class] = False

                error = np.sum(class_counts[inverse_index])

                # compute the total error for each feature and
                #  save all the corresponding rules for a given feature
                if feature_index not in prediction_dict:
                    prediction_dict[feature_index] = {
                        "total error": 0,
                        "rules (value: class)": {},
                    }
                prediction_dict[feature_index]["rules (value: class)"][
                    feature_value
                ] = most_frequent_class
                prediction_dict[feature_index]["total error"] += error

            # get best feature (i.e., the feature with the lowest error)
            best_err = np.inf
            best_idx = [None]
            for i in prediction_dict:
                if prediction_dict[i]["total error"] < best_err:
                    best_err = prediction_dict[i]["total error"]
                    best_idx[-1] = i

            if self.resolve_ties == "chi-squared":
                # collect duplicates
                for i in prediction_dict:
                    if i == best_idx[-1]:
                        continue
                    if prediction_dict[i]["total error"] == best_err:
                        best_idx.append(i)

                p_values = []
                for feature_idx in best_idx:
                    rules = prediction_dict[feature_idx]["rules (value: class)"]

                    # contingency table for a given feature
                    #   (e.g., petal_width for iris)
                    #   is organized as follows (without the sum columns):
                    #
                    #              petal_width
                    # species      (0.0976,0.791] (0.791,1.63] (1.63,2.5] sum
                    #   setosa                 50            0          0  50
                    #   versicolor              0           48          2  50
                    #   virginica               0            4         46  50
                    #   sum                    50           52         48 150

                    ary = np.zeros((n_class_labels, len(rules)))

                    for idx, r in enumerate(rules):
                        ary[:, idx] = np.bincount(
                            y[X[:, feature_idx] == r], minlength=n_class_labels
                        )

                    # returns "stat, p, dof, expected"
                    _, p, _, _ = chi2_contingency(ary)
                p_values.append(p)
                best_p_idx = np.argmax(p_values)
                best_idx = best_idx[best_p_idx]
                self.p_value_ = p_values[best_p_idx]

            elif self.resolve_ties == "first":
                best_idx = best_idx[0]

        self.feature_idx_ = best_idx
        self.prediction_dict_ = prediction_dict[best_idx]
        return self

    def predict(self, X):
        """Predict class labels for X.

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
        if not hasattr(self, "prediction_dict_"):
            raise NotFittedError(
                "Estimator not fitted, " "call `fit` before using the model."
            )

        rules = self.prediction_dict_["rules (value: class)"]

        y_pred = np.zeros(X.shape[0], dtype=np.int_)

        # Set up labels for those class labels in the
        # dataset for which no rule exists. We use the
        # first non-specified class label as the default class label.
        rule_labels = set()
        for feature_value in rules:
            class_label = rules[feature_value]
            rule_labels.add(class_label)
        other_label = set(self.class_labels_) - rule_labels
        if len(other_label):
            y_pred[:] = list(other_label)[0]
        # else just use "np.zeros"; we could also change this to
        #  self.class_labels_[-1]+1 in future

        # classify all class labels for which rules exist
        for feature_value in rules:
            mask = X[:, self.feature_idx_] == feature_value
            y_pred[mask] = rules[feature_value]

        return y_pred
