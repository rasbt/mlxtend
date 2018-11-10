## feature_importance_permutation

*feature_importance_permutation(X, y, predict_method, metric, num_rounds=1, seed=None)*

Feature importance imputation via permutation importance

**Parameters**


- `X` : NumPy array, shape = [n_samples, n_features]

    Dataset, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : NumPy array, shape = [n_samples]

    Target values.


- `predict_method` : prediction function

    A callable function that predicts the target values
    from X.


- `metric` : str, callable

    The metric for evaluating the feature importance through
    permutation. By default, the strings 'accuracy' is
    recommended for classifiers and the string 'r2' is
    recommended for regressors. Optionally, a custom
    scoring function (e.g., `metric=scoring_func`) that
    accepts two arguments, y_true and y_pred, which have
    similar shape to the `y` array.


- `num_rounds` : int (default=1)

    Number of rounds the feature columns are permuted to
    compute the permutation importance.


- `seed` : int or None (default=None)

    Random seed for permuting the feature columns.

**Returns**


- `mean_importance_vals, all_importance_vals` : NumPy arrays.

    The first array, mean_importance_vals has shape [n_features, ] and
    contains the importance values for all features.
    The shape of the second array is [n_features, num_rounds] and contains
    the feature importance for each repetition. If num_rounds=1,
    it contains the same values as the first array, mean_importance_vals.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/](http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/)

