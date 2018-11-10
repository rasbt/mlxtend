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

