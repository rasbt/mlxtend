# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Feature Importance Estimation Through Permutation
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def feature_importance_permutation(
    X, y, predict_method, metric, num_rounds=1, feature_groups=None, seed=None
):
    """Feature importance imputation via permutation importance

    Parameters
    ----------

    X : NumPy array, shape = [n_samples, n_features]
        Dataset, where n_samples is the number of samples and
        n_features is the number of features.

    y : NumPy array, shape = [n_samples]
        Target values.

    predict_method : prediction function
        A callable function that predicts the target values
        from X.

    metric : str, callable
        The metric for evaluating the feature importance through
        permutation. By default, the strings 'accuracy' is
        recommended for classifiers and the string 'r2' is
        recommended for regressors. Optionally, a custom
        scoring function (e.g., `metric=scoring_func`) that
        accepts two arguments, y_true and y_pred, which have
        similar shape to the `y` array.

    num_rounds : int (default=1)
        Number of rounds the feature columns are permuted to
        compute the permutation importance.

    feature_groups : list or None (default=None)
        Optional argument for treating certain features as a group.
        For example `[1, 2, [3, 4, 5]]`, which can be useful for
        interpretability, for example, if features 3, 4, 5 are one-hot
        encoded features.

    seed : int or None (default=None)
        Random seed for permuting the feature columns.

    Returns
    ---------

    mean_importance_vals, all_importance_vals : NumPy arrays.
      The first array, mean_importance_vals has shape [n_features, ] and
      contains the importance values for all features.
      The shape of the second array is [n_features, num_rounds] and contains
      the feature importance for each repetition. If num_rounds=1,
      it contains the same values as the first array, mean_importance_vals.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/

    """

    if not isinstance(num_rounds, int):
        raise ValueError("num_rounds must be an integer.")
    if num_rounds < 1:
        raise ValueError("num_rounds must be greater than 1.")

    if not (metric in ("r2", "accuracy") or hasattr(metric, "__call__")):
        raise ValueError(
            'metric must be either "r2", "accuracy", '
            "or a function with signature func(y_true, y_pred)."
        )

    if metric == "r2":

        def score_func(y_true, y_pred):
            sum_of_squares = np.sum(np.square(y_true - y_pred))
            res_sum_of_squares = np.sum(np.square(y_true - y_true.mean()))
            r2_score = 1.0 - (sum_of_squares / res_sum_of_squares)
            return r2_score

    elif metric == "accuracy":

        def score_func(y_true, y_pred):
            return np.mean(y_true == y_pred)

    else:
        score_func = metric

    rng = np.random.RandomState(seed)

    baseline = score_func(y, predict_method(X))

    if feature_groups is None:
        mean_importance_vals = np.zeros(X.shape[1])
        all_importance_vals = np.zeros((X.shape[1], num_rounds))

        for round_idx in range(num_rounds):
            for col_idx in range(X.shape[1]):
                save_col = X[:, col_idx].copy()
                rng.shuffle(X[:, col_idx])
                new_score = score_func(y, predict_method(X))
                X[:, col_idx] = save_col
                importance = baseline - new_score
                mean_importance_vals[col_idx] += importance
                all_importance_vals[col_idx, round_idx] = importance
        mean_importance_vals /= num_rounds

    else:
        mean_importance_vals = np.zeros(len(feature_groups))
        all_importance_vals = np.zeros((len(feature_groups), num_rounds))
        for round_idx in range(num_rounds):
            for col_idx, feat in enumerate(feature_groups):
                save_col = X[:, feat].copy()

                if save_col.ndim > 1:
                    columns = save_col.shape[1]
                    for i in range(columns):
                        rng.shuffle(X[:, i])
                else:
                    rng.shuffle(X[:, feat])

                new_score = score_func(y, predict_method(X))
                X[:, feat] = save_col
                importance = baseline - new_score
                mean_importance_vals[col_idx] += importance
                all_importance_vals[col_idx, round_idx] = importance
        mean_importance_vals /= num_rounds

    return mean_importance_vals, all_importance_vals
