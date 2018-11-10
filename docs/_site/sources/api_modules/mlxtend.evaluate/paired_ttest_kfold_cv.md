## paired_ttest_kfold_cv

*paired_ttest_kfold_cv(estimator1, estimator2, X, y, cv=10, scoring=None, shuffle=False, random_seed=None)*

Implements the k-fold paired t test procedure
to compare the performance of two models.

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


- `cv` : int (default: 10)

    Number of splits and iteration for the
    cross-validation procedure


- `scoring` : str, callable, or None (default: None)

    If None (default), uses 'accuracy' for sklearn classifiers
    and 'r2' for sklearn regressors.
    If str, uses a sklearn scoring metric string identifier, for example
    {accuracy, f1, precision, recall, roc_auc} for classifiers,
    {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
    'median_absolute_error', 'r2'} for regressors.
    If a callable object or function is provided, it has to be conform with
    sklearn's signature ``scorer(estimator, X, y)``; see
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    for more information.


- `shuffle` : bool (default: True)

    Whether to shuffle the dataset for generating
    the k-fold splits.


- `random_seed` : int or None (default: None)

    Random seed for shuffling the dataset
    for generating the k-fold splits.
    Ignored if shuffle=False.

**Returns**

- `t` : float

    The t-statistic


- `pvalue` : float

    Two-tailed p-value.
    If the chosen significance level is larger
    than the p-value, we reject the null hypothesis
    and accept that there are significant differences
    in the two compared models.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_kfold_cv/](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_kfold_cv/)

