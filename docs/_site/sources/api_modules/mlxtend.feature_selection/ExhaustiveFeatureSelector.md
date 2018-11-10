## ExhaustiveFeatureSelector

*ExhaustiveFeatureSelector(estimator, min_features=1, max_features=1, print_progress=True, scoring='accuracy', cv=5, n_jobs=1, pre_dispatch='2*n_jobs', clone_estimator=True)*

Exhaustive Feature Selection for Classification and Regression.
(new in v0.4.3)

**Parameters**

- `estimator` : scikit-learn classifier or regressor


- `min_features` : int (default: 1)

    Minumum number of features to select

- `max_features` : int (default: 1)

    Maximum number of features to select

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

- `n_jobs` : int (default: 1)

    The number of CPUs to use for evaluating different feature subsets
    in parallel. -1 means 'all CPUs'.

- `pre_dispatch` : int, or string (default: '2*n_jobs')

    Controls the number of jobs that get dispatched
    during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
    Reducing this number can be useful to avoid an explosion of
    memory consumption when more jobs get dispatched than CPUs can process.
    This parameter can be:
    None, in which case all the jobs are immediately created and spawned.
    Use this for lightweight and fast-running jobs,
    to avoid delays due to on-demand spawning of the jobs
    An int, giving the exact number of total jobs that are spawned
    A string, giving an expression as a function
    of n_jobs, as in `2*n_jobs`

- `clone_estimator` : bool (default: True)

    Clones estimator if True; works with the original estimator instance
    if False. Set to False if the estimator doesn't
    implement scikit-learn's set_params and get_params methods.
    In addition, it is required to set cv=0, and n_jobs=1.

**Attributes**

- `best_idx_` : array-like, shape = [n_predictions]

    Feature Indices of the selected feature subsets.

- `best_feature_names_` : array-like, shape = [n_predictions]

    Feature names of the selected feature subsets. If pandas
    DataFrames are used in the `fit` method, the feature
    names correspond to the column names. Otherwise, the
    feature names are string representation of the feature
    array indices. New in v 0.13.0.

- `best_score_` : float

    Cross validation average score of the selected subset.

- `subsets_` : dict

    A dictionary of selected feature subsets during the
    exhaustive selection, where the dictionary keys are
    the lengths k of these feature subsets. The dictionary
    values are dictionaries themselves with the following
    keys: 'feature_idx' (tuple of indices of the feature subset)
    'feature_names' (tuple of feature names of the feat. subset)
    'cv_scores' (list individual cross-validation scores)
    'avg_score' (average cross-validation score)
    Note that if pandas
    DataFrames are used in the `fit` method, the 'feature_names'
    correspond to the column names. Otherwise, the
    feature names are string representation of the feature
    array indices. The 'feature_names' is new in v 0.13.0.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/)

### Methods

<hr>

*fit(X, y, custom_feature_names=None, **fit_params)*

Perform feature selection and learn model from training data.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

- `y` : array-like, shape = [n_samples]

    Target values.

- `custom_feature_names` : None or tuple (default: tuple)

    Custom feature names for `self.k_feature_names` and
    `self.subsets_[i]['feature_names']`.
    (new in v 0.13.0)

- `fit_params` : dict of string -> object, optional

    Parameters to pass to to the fit method of classifier.

**Returns**

- `self` : object


<hr>

*fit_transform(X, y, **fit_params)*

Fit to training data and return the best selected features from X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

- `y` : array-like, shape = [n_samples]

    Target values.

- `fit_params` : dict of string -> object, optional

    Parameters to pass to to the fit method of classifier.

**Returns**

Feature subset of X, shape={n_samples, k_features}

<hr>

*get_metric_dict(confidence_interval=0.95)*

Return metric dictionary

**Parameters**

- `confidence_interval` : float (default: 0.95)

    A positive float between 0.0 and 1.0 to compute the confidence
    interval bounds of the CV score averages.

**Returns**

Dictionary with items where each dictionary value is a list
    with the number of iterations (number of feature subsets) as
    its length. The dictionary keys corresponding to these lists
    are as follows:
    'feature_idx': tuple of the indices of the feature subset
    'cv_scores': list with individual CV scores
    'avg_score': of CV average scores
    'std_dev': standard deviation of the CV score average
    'std_err': standard error of the CV score average
    'ci_bound': confidence interval bound of the CV score average

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X)*

Return the best selected features from X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

**Returns**

Feature subset of X, shape={n_samples, k_features}

