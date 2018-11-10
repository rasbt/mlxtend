## paired_ttest_resampled

*paired_ttest_resampled(estimator1, estimator2, X, y, num_rounds=30, test_size=0.3, scoring=None, random_seed=None)*

Implements the resampled paired t test procedure
to compare the performance of two models
(also called k-hold-out paired t test).

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


- `num_rounds` : int (default: 30)

    Number of resampling iterations
    (i.e., train/test splits)


- `test_size` : float or int (default: 0.3)

    If float, should be between 0.0 and 1.0 and
    represent the proportion of the dataset to use
    as a test set.
    If int, represents the absolute number of test exsamples.


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


- `random_seed` : int or None (default: None)

    Random seed for creating the test/train splits.

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
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/)

