## SequentialFeatureSelector

*SequentialFeatureSelector(estimator, k_features, forward=True, floating=False, print_progress=True, scoring='accuracy', cv=5, skip_if_stuck=True, n_jobs=1, pre_dispatch='2*n_jobs')*

Sequential Feature Selection for Classification and Regression.

**Parameters**

- `estimator` : scikit-learn classifier or regressor


- `k_features` : int

    Number of features to select,
    where k_features < the full feature set.

- `forward` : bool (default: True)

    Forward selection if True,
    backward selection otherwise

- `floating` : bool (default: False)

    Adds a conditional exclusion/inclusion if True.

- `print_progress` : bool (default: True)

    Prints progress as the number of epochs
    to stderr.

- `scoring` : str, (default='accuracy')

    Scoring metric in {accuracy, f1, precision, recall, roc_auc}
    for classifiers,
    {'mean_absolute_error', 'mean_squared_error',
    'median_absolute_error', 'r2'} for regressors,
    or a callable object or function with
    signature ``scorer(estimator, X, y)``.

- `cv` : int (default: 5)

    Scikit-learn cross-validation generator or `int`.
    If estimator is a classifier (or y consists of integer class labels),
    stratified k-fold is performed, and regular k-fold cross-validation
    otherwise.
    No cross-validation if cv is None, False, or 0.
    skip_if_stuck: bool (default: True)
    Set to True to skip conditional
    exlusion/inclusion if floating=True and
    algorithm gets stuck in cycles.

- `n_jobs` : int (default: 1)

    The number of CPUs to use for cross validation. -1 means 'all CPUs'.

- `pre_dispatch` : int, or string

    Controls the number of jobs that get dispatched
    during parallel execution in cross_val_score.
    Reducing this number can be useful to avoid an explosion of
    memory consumption when more jobs get dispatched than CPUs can process.
    This parameter can be:
    None, in which case all the jobs are immediately created and spawned.
    Use this for lightweight and fast-running jobs,
    to avoid delays due to on-demand spawning of the jobs
    An int, giving the exact number of total jobs that are spawned
    A string, giving an expression as a function
    of n_jobs, as in `2*n_jobs`

**Attributes**

- `k_feature_idx_` : array-like, shape = [n_predictions]

    Feature Indices of the selected feature subsets.

- `k_score_` : float

    Cross validation average score of the selected subset.

- `subsets_` : dict

    A dictionary of selected feature subsets during the
    sequential selection, where the dictionary keys are
    the lenghts k of these feature subsets. The dictionary
    values are dictionaries themselves with the following
    keys: 'feature_idx' (tuple of indices of the feature subset)
    'cv_scores' (list individual cross-validation scores)
    'avg_score' (average cross-validation score)

**Examples**
    >>> from sklearn.neighbors import KNeighborsClassifier

    >>> from sklearn.datasets import load_iris

    >>> iris = load_iris()

    >>> X = iris.data

    >>> y = iris.target

    >>> knn = KNeighborsClassifier(n_neighbors=4)

    >>> sfs = SequentialFeatureSelector(knn, k_features=2,

    ...                                 scoring='accuracy', cv=5)
    >>> sfs = sfs.fit(X, y)

    >>> sfs.indices_

    (2, 3)
    >>> sfs.transform(X[:5])

    array([[ 1.4,  0.2],
    [ 1.4,  0.2],
    [ 1.3,  0.2],
    [ 1.5,  0.2],
    [ 1.4,  0.2]])

    >>> print('best score: %.2f' % sfs.k_score_)

    best score: 0.97

### Methods

<hr>

*fit(X, y)*

None

<hr>

*fit_transform(X, y)*

None

<hr>

*get_metric_dict(confidence_interval=0.95)*

None

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**
    deep: boolean, optional
    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

    The method works on simple estimators as well as on nested objects
    (such as pipelines). The former have parameters of the form
    ``<component>__<parameter>`` so that it's possible to update each
    component of a nested object.

**Returns**
    self

<hr>

*transform(X)*

None

