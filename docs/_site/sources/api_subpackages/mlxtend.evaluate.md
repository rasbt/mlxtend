mlxtend version: 0.15.0dev 
## BootstrapOutOfBag

*BootstrapOutOfBag(n_splits=200, random_seed=None)*

**Parameters**


- `n_splits` : int (default=200)

    Number of bootstrap iterations.
    Must be larger than 1.


- `random_seed` : int (default=None)

    If int, random_seed is the seed used by
    the random number generator.


**Returns**

- `train_idx` : ndarray

    The training set indices for that split.


- `test_idx` : ndarray

    The testing set indices for that split.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/](http://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/)

### Methods

<hr>

*get_n_splits(X=None, y=None, groups=None)*

Returns the number of splitting iterations in the cross-validator

**Parameters**

- `X` : object

    Always ignored, exists for compatibility with scikit-learn.


- `y` : object

    Always ignored, exists for compatibility with scikit-learn.


- `groups` : object

    Always ignored, exists for compatibility with scikit-learn.

**Returns**


- `n_splits` : int

    Returns the number of splitting iterations in the cross-validator.

<hr>

*split(X, y=None, groups=None)*

y : array-like or None (default: None)
Argument is not used and only included as parameter
for compatibility, similar to `KFold` in scikit-learn.


- `groups` : array-like or None (default: None)

    Argument is not used and only included as parameter
    for compatibility, similar to `KFold` in scikit-learn.




## PredefinedHoldoutSplit

*PredefinedHoldoutSplit(valid_indices)*

Train/Validation set splitter for sklearn's GridSearchCV etc.

Uses user-specified train/validation set indices to split a dataset
into train/validation sets using user-defined or random
indices.

**Parameters**

- `valid_indices` : array-like, shape (num_examples,)

    Indices of the training examples in the training set
    to be used for validation. All other indices in the
    training set are used to for a training subset
    for model fitting.

### Methods

<hr>

*get_n_splits(X=None, y=None, groups=None)*

Returns the number of splitting iterations in the cross-validator

**Parameters**

- `X` : object

    Always ignored, exists for compatibility.


- `y` : object

    Always ignored, exists for compatibility.


- `groups` : object

    Always ignored, exists for compatibility.

**Returns**

- `n_splits` : 1

    Returns the number of splitting iterations in the cross-validator.
    Always returns 1.

<hr>

*split(X, y, groups=None)*

Generate indices to split data into training and test set.

**Parameters**

- `X` : array-like, shape (num_examples, num_features)

    Training data, where num_examples is the number of examples
    and num_features is the number of features.


- `y` : array-like, shape (num_examples,)

    The target variable for supervised learning problems.
    Stratification is done based on the y labels.


- `groups` : object

    Always ignored, exists for compatibility.

**Yields**

- `train_index` : ndarray

    The training set indices for that split.


- `valid_index` : ndarray

    The validation set indices for that split.




## RandomHoldoutSplit

*RandomHoldoutSplit(valid_size=0.5, random_seed=None, stratify=False)*

Train/Validation set splitter for sklearn's GridSearchCV etc.

Provides train/validation set indices to split a dataset
into train/validation sets using random indices.

**Parameters**

- `valid_size` : float (default: 0.5)

    Proportion of examples that being assigned as
    validation examples. 1-`valid_size` will then automatically
    be assigned as training set examples.

- `random_seed` : int (default: None)

    The random seed for splitting the data
    into training and validation set partitions.

- `stratify` : bool (default: False)

    True or False, whether to perform a stratified
    split or not

### Methods

<hr>

*get_n_splits(X=None, y=None, groups=None)*

Returns the number of splitting iterations in the cross-validator

**Parameters**

- `X` : object

    Always ignored, exists for compatibility.


- `y` : object

    Always ignored, exists for compatibility.


- `groups` : object

    Always ignored, exists for compatibility.

**Returns**

- `n_splits` : 1

    Returns the number of splitting iterations in the cross-validator.
    Always returns 1.

<hr>

*split(X, y, groups=None)*

Generate indices to split data into training and test set.

**Parameters**

- `X` : array-like, shape (num_examples, num_features)

    Training data, where num_examples is the number of
    training examples and num_features is the number of features.


- `y` : array-like, shape (num_examples,)

    The target variable for supervised learning problems.
    Stratification is done based on the y labels.


- `groups` : object

    Always ignored, exists for compatibility.

**Yields**

- `train_index` : ndarray

    The training set indices for that split.


- `valid_index` : ndarray

    The validation set indices for that split.




## bias_variance_decomp

*bias_variance_decomp(estimator, X_train, y_train, X_test, y_test, loss='0-1_loss', num_rounds=200, random_seed=None)*

estimator : object
A classifier or regressor object or class implementing a `fit`
`predict` method similar to the scikit-learn API.


- `X_train` : array-like, shape=(num_examples, num_features)

    A training dataset for drawing the bootstrap samples to carry
    out the bias-variance decomposition.


- `y_train` : array-like, shape=(num_examples)

    Targets (class labels, continuous values in case of regression)
    associated with the `X_train` examples.


- `X_test` : array-like, shape=(num_examples, num_features)

    The test dataset for computing the average loss, bias,
    and variance.


- `y_test` : array-like, shape=(num_examples)

    Targets (class labels, continuous values in case of regression)
    associated with the `X_test` examples.


- `loss` : str (default='0-1_loss')

    Loss function for performing the bias-variance decomposition.
    Currently allowed values are '0-1_loss' and 'mse'.


- `num_rounds` : int (default=200)

    Number of bootstrap rounds for performing the bias-variance
    decomposition.


- `random_seed` : int (default=None)

    Random seed for the bootstrap sampling used for the
    bias-variance decomposition.

**Returns**

- `avg_expected_loss, avg_bias, avg_var` : returns the average expected

    average bias, and average bias (all floats), where the average
    is computed over the data points in the test set.




## bootstrap

*bootstrap(x, func, num_rounds=1000, ci=0.95, ddof=1, seed=None)*

Implements the ordinary nonparametric bootstrap

**Parameters**


- `x` : NumPy array, shape=(n_samples, [n_columns])

    An one or multidimensional array of data records


- `func` : <func>

    A function which computes a statistic that is used
    to compute the bootstrap replicates (the statistic computed
    from the bootstrap samples). This function must return a
    scalar value. For example, `np.mean` or `np.median` would be
    an acceptable argument for `func` if `x` is a 1-dimensional array
    or vector.


- `num_rounds` : int (default=1000)

    The number of bootstrap samnples to draw where each
    bootstrap sample has the same number of records as the
    original dataset.


- `ci` : int (default=0.95)

    An integer in the range (0, 1) that represents the
    confidence level for computing the confidence interval.
    For example, `ci=0.95` (default)
    will compute the 95% confidence
    interval from the bootstrap replicates.


- `ddof` : int

    The delta degrees of freedom used when computing the
    standard error.


- `seed` : int or None (default=None)

    Random seed for generating bootstrap samples.

**Returns**


- `original, standard_error, (lower_ci, upper_ci)` : tuple

    Returns the statistic of the original sample (`original`),
    the standard error of the estimate, and the
    respective confidence interval bounds.

**Examples**

    >>> from mlxtend.evaluate import bootstrap
    >>> rng = np.random.RandomState(123)
    >>> x = rng.normal(loc=5., size=100)
    >>> original, std_err, ci_bounds = bootstrap(x,
    ...                                          num_rounds=1000,
    ...                                          func=np.mean,
    ...                                          ci=0.95,
    ...                                          seed=123)
    >>> print('Mean: %.2f, SE: +/- %.2f, CI95: [%.2f, %.2f]' % (original,
    ...                                                         std_err,
    ...                                                         ci_bounds[0],
    ...                                                         ci_bounds[1]))
    Mean: 5.03, SE: +/- 0.11, CI95: [4.80, 5.26]
    >>>

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap/](http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap/)




## bootstrap_point632_score

*bootstrap_point632_score(estimator, X, y, n_splits=200, method='.632', scoring_func=None, random_seed=None, clone_estimator=True)*

Implementation of the .632 [1] and .632+ [2] bootstrap
for supervised learning

References:

- [1] Efron, Bradley. 1983. "Estimating the Error Rate
of a Prediction Rule: Improvement on Cross-Validation."
Journal of the American Statistical Association
78 (382): 316. doi:10.2307/2288636.
- [2] Efron, Bradley, and Robert Tibshirani. 1997.
"Improvements on Cross-Validation: The .632+ Bootstrap Method."
Journal of the American Statistical Association
92 (438): 548. doi:10.2307/2965703.

**Parameters**

- `estimator` : object

    An estimator for classification or regression that
    follows the scikit-learn API and implements "fit" and "predict"
    methods.


- `X` : array-like

    The data to fit. Can be, for example a list, or an array at least 2d.


- `y` : array-like, optional, default: None

    The target variable to try to predict in the case of
    supervised learning.


- `n_splits` : int (default=200)

    Number of bootstrap iterations.
    Must be larger than 1.


- `method` : str (default='.632')

    The bootstrap method, which can be either
    - 1) '.632' bootstrap (default)
    - 2) '.632+' bootstrap
    - 3) 'oob' (regular out-of-bag, no weighting)
    for comparison studies.


- `scoring_func` : callable,

    Score function (or loss function) with signature
``scoring_func(y, y_pred, **kwargs)``.
    If none, uses classification accuracy if the

estimator is a classifier and mean squared error
    if the estimator is a regressor.


- `random_seed` : int (default=None)

    If int, random_seed is the seed used by
    the random number generator.


- `clone_estimator` : bool (default=True)

    Clones the estimator if true, otherwise fits
    the original.

**Returns**

- `scores` : array of float, shape=(len(list(n_splits)),)

    Array of scores of the estimator for each bootstrap
    replicate.

**Examples**


    >>> from sklearn import datasets, linear_model
    >>> from mlxtend.evaluate import bootstrap_point632_score
    >>> iris = datasets.load_iris()
    >>> X = iris.data
    >>> y = iris.target
    >>> lr = linear_model.LogisticRegression()
    >>> scores = bootstrap_point632_score(lr, X, y)
    >>> acc = np.mean(scores)
    >>> print('Accuracy:', acc)
    0.953023146884
    >>> lower = np.percentile(scores, 2.5)
    >>> upper = np.percentile(scores, 97.5)
    >>> print('95%% Confidence interval: [%.2f, %.2f]' % (lower, upper))
    95% Confidence interval: [0.90, 0.98]

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/](http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap_point632_score/)




## cochrans_q

*cochrans_q(y_target, *y_model_predictions)*

Cochran's Q test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `q, p` : float or None, float

    Returns the Q (chi-squared) value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/](http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/)




## combined_ftest_5x2cv

*combined_ftest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None)*

Implements the 5x2cv combined F test proposed
by Alpaydin 1999,
to compare the performance of two models.

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


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

- `f` : float

    The F-statistic


- `pvalue` : float

    Two-tailed p-value.
    If the chosen significance level is larger
    than the p-value, we reject the null hypothesis
    and accept that there are significant differences
    in the two compared models.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/](http://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/)




## confusion_matrix

*confusion_matrix(y_target, y_predicted, binary=False, positive_label=1)*

Compute a confusion matrix/contingency table.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels.

- `y_predicted` : array-like, shape=[n_samples]

    Predicted class labels.

- `binary` : bool (default: False)

    Maps a multi-class problem onto a
    binary confusion matrix, where
    the positive class is 1 and
    all other classes are 0.

- `positive_label` : int (default: 1)

    Class label of the positive class.

**Returns**

- `mat` : array-like, shape=[n_classes, n_classes]


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/](http://rasbt.github.io/mlxtend/user_guide/evaluate/confusion_matrix/)




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




## ftest

*ftest(y_target, *y_model_predictions)*

F-Test test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `f, p` : float or None, float

    Returns the F-value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/](http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/)




## lift_score

*lift_score(y_target, y_predicted, binary=True, positive_label=1)*

Lift measures the degree to which the predictions of a
classification model are better than randomly-generated predictions.

The in terms of True Positives (TP), True Negatives (TN),
False Positives (FP), and False Negatives (FN), the lift score is
computed as:
[ TP / (TP+FP) ] / [ (TP+FN) / (TP+TN+FP+FN) ]


**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels.

- `y_predicted` : array-like, shape=[n_samples]

    Predicted class labels.

- `binary` : bool (default: True)

    Maps a multi-class problem onto a
    binary, where
    the positive class is 1 and
    all other classes are 0.

- `positive_label` : int (default: 0)

    Class label of the positive class.

**Returns**

- `score` : float

    Lift score in the range [0, $\infty$]

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/](http://rasbt.github.io/mlxtend/user_guide/evaluate/lift_score/)




## mcnemar

*mcnemar(ary, corrected=True, exact=False)*

McNemar test for paired nominal data

**Parameters**

- `ary` : array-like, shape=[2, 2]

    2 x 2 contigency table (as returned by evaluate.mcnemar_table),
    where
    a: ary[0, 0]: # of samples that both models predicted correctly
    b: ary[0, 1]: # of samples that model 1 got right and model 2 got wrong
    c: ary[1, 0]: # of samples that model 2 got right and model 1 got wrong
    d: aryCell [1, 1]: # of samples that both models predicted incorrectly

- `corrected` : array-like, shape=[n_samples] (default: True)

    Uses Edward's continuity correction for chi-squared if `True`

- `exact` : bool, (default: False)

    If `True`, uses an exact binomial test comparing b to
    a binomial distribution with n = b + c and p = 0.5.
    It is highly recommended to use `exact=True` for sample sizes < 25
    since chi-squared is not well-approximated
    by the chi-squared distribution!

**Returns**

- `chi2, p` : float or None, float

    Returns the chi-squared value and the p-value;
    if `exact=True` (default: `False`), `chi2` is `None`

**Examples**

    For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar/)




## mcnemar_table

*mcnemar_table(y_target, y_model1, y_model2)*

Compute a 2x2 contigency table for McNemar's test.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.

- `y_model1` : array-like, shape=[n_samples]

    Predicted class labels from model as 1D NumPy array.

- `y_model2` : array-like, shape=[n_samples]

    Predicted class labels from model 2 as 1D NumPy array.

**Returns**

- `tb` : array-like, shape=[2, 2]

    2x2 contingency table with the following contents:
    a: tb[0, 0]: # of samples that both models predicted correctly
    b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
    c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
    d: tb[1, 1]: # of samples that both models predicted incorrectly

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/)




## mcnemar_tables

*mcnemar_tables(y_target, *y_model_predictions)*

Compute multiple 2x2 contigency tables for McNemar's
test or Cochran's Q test.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `y_model_predictions` : array-like, shape=[n_samples]

    Predicted class labels for a model.

**Returns**


- `tables` : dict

    Dictionary of NumPy arrays with shape=[2, 2]. Each dictionary
    key names the two models to be compared based on the order the
    models were passed as `*y_model_predictions`. The number of
    dictionary entries is equal to the number of pairwise combinations
    between the m models, i.e., "m choose 2."

    For example the following target array (containing the true labels)
    and 3 models

    - y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    - y_mod0 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    - y_mod0 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    - y_mod0 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    would result in the following dictionary:


    {'model_0 vs model_1': array([[ 4.,  1.],
    [ 2.,  3.]]),
    'model_0 vs model_2': array([[ 3.,  0.],
    [ 3.,  4.]]),
    'model_1 vs model_2': array([[ 3.,  0.],
    [ 2.,  5.]])}

    Each array is structured in the following way:

    - tb[0, 0]: # of samples that both models predicted correctly
    - tb[0, 1]: # of samples that model a got right and model b got wrong
    - tb[1, 0]: # of samples that model b got right and model a got wrong
    - tb[1, 1]: # of samples that both models predicted incorrectly

**Examples**

    For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/)




## paired_ttest_5x2cv

*paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None)*

Implements the 5x2cv paired t test proposed
by Dieterrich (1998)
to compare the performance of two models.

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


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
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/)




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




## permutation_test

*permutation_test(x, y, func='x_mean != y_mean', method='exact', num_rounds=1000, seed=None)*

Nonparametric permutation test

**Parameters**

- `x` : list or numpy array with shape (n_datapoints,)

    A list or 1D numpy array of the first sample
    (e.g., the treatment group).

- `y` : list or numpy array with shape (n_datapoints,)

    A list or 1D numpy array of the second sample
    (e.g., the control group).

- `func` : custom function or str (default: 'x_mean != y_mean')

    function to compute the statistic for the permutation test.
    - If 'x_mean != y_mean', uses
    `func=lambda x, y: np.abs(np.mean(x) - np.mean(y)))`
    for a two-sided test.
    - If 'x_mean > y_mean', uses
    `func=lambda x, y: np.mean(x) - np.mean(y))`
    for a one-sided test.
    - If 'x_mean < y_mean', uses
    `func=lambda x, y: np.mean(y) - np.mean(x))`
    for a one-sided test.

- `method` : 'approximate' or 'exact' (default: 'exact')

    If 'exact' (default), all possible permutations are considered.
    If 'approximate' the number of drawn samples is
    given by `num_rounds`.
    Note that 'exact' is typically not feasible unless the dataset
    size is relatively small.

- `num_rounds` : int (default: 1000)

    The number of permutation samples if `method='approximate'`.

- `seed` : int or None (default: None)

    The random seed for generating permutation samples if
    `method='approximate'`.

**Returns**

p-value under the null hypothesis

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/](http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/)




## proportion_difference

*proportion_difference(proportion_1, proportion_2, n_1, n_2=None)*

Computes the test statistic and p-value for a difference of
proportions test.

**Parameters**

- `proportion_1` : float

    The first proportion

- `proportion_2` : float

    The second proportion

- `n_1` : int

    The sample size of the first test sample

- `n_2` : int or None (default=None)

    The sample size of the second test sample.
    If `None`, `n_1`=`n_2`.

**Returns**


- `z, p` : float or None, float

    Returns the z-score and the p-value


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/](http://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/)




## scoring

*scoring(y_target, y_predicted, metric='error', positive_label=1, unique_labels='auto')*

Compute a scoring metric for supervised learning.

**Parameters**

- `y_target` : array-like, shape=[n_values]

    True class labels or target values.

- `y_predicted` : array-like, shape=[n_values]

    Predicted class labels or target values.

- `metric` : str (default: 'error')

    Performance metric:
    'accuracy': (TP + TN)/(FP + FN + TP + TN) = 1-ERR

    'per-class accuracy': Average per-class accuracy

    'per-class error':  Average per-class error

    'error': (TP + TN)/(FP+ FN + TP + TN) = 1-ACC

    'false_positive_rate': FP/N = FP/(FP + TN)

    'true_positive_rate': TP/P = TP/(FN + TP)

    'true_negative_rate': TN/N = TN/(FP + TN)

    'precision': TP/(TP + FP)

    'recall': equal to 'true_positive_rate'

    'sensitivity': equal to 'true_positive_rate' or 'recall'

    'specificity': equal to 'true_negative_rate'

    'f1': 2 * (PRE * REC)/(PRE + REC)

    'matthews_corr_coef':  (TP*TN - FP*FN)
    / (sqrt{(TP + FP)( TP + FN )( TN + FP )( TN + FN )})

    Where:
    [TP: True positives, TN = True negatives,

    TN: True negatives, FN = False negatives]


- `positive_label` : int (default: 1)

    Label of the positive class for binary classification
    metrics.

- `unique_labels` : str or array-like (default: 'auto')

    If 'auto', deduces the unique class labels from
    y_target

**Returns**

- `score` : float


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/scoring/](http://rasbt.github.io/mlxtend/user_guide/evaluate/scoring/)




