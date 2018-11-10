mlxtend version: 0.14.0dev 
## LinearRegression

*LinearRegression(eta=0.01, epochs=50, minibatches=None, random_seed=None, print_progress=0)*

Ordinary least squares linear regression.

**Parameters**

- `eta` : float (default: 0.01)

    solver rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Passes over the training dataset.
    Prior to each epoch, the dataset is shuffled
    if `minibatches > 1` to prevent cycles in stochastic gradient descent.

- `minibatches` : int (default: None)

    The number of minibatches for gradient-based optimization.
    If None: Normal Equations (closed-form solution)
    If 1: Gradient Descent learning
    If len(y): Stochastic Gradient Descent learning
    If 1 < minibatches < len(y): Minibatch learning

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr if not solver='normal equation'
    0: No output
    1: Epochs elapsed and cost
    2: 1 plus time elapsed
    3: 2 plus estimated time until completion

**Attributes**

- `w_` : 2d-array, shape={n_features, 1}

    Model weights after fitting.

- `b_` : 1d-array, shape={1,}

    Bias unit after fitting.

- `cost_` : list

    Sum of squared errors after each epoch;
    ignored if solver='normal equation'

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/](http://rasbt.github.io/mlxtend/user_guide/regressor/LinearRegression/)

### Methods

<hr>

*fit(X, y, init_params=True)*

Learn model from training data.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values.

- `init_params` : bool (default: True)

    Re-initializes model parameters prior to fitting.
    Set False to continue training with weights from
    a previous model fitting.

**Returns**

- `self` : object


<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.'

    adapted from
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
    # Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
    # License: BSD 3 clause

<hr>

*predict(X)*

Predict targets from X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `target_values` : array-like, shape = [n_samples]

    Predicted target values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.
The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

adapted from
https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
# License: BSD 3 clause




## StackingCVRegressor

*StackingCVRegressor(regressors, meta_regressor, cv=5, shuffle=True, use_features_in_secondary=False, store_train_meta_features=False, refit=True)*

A 'Stacking Cross-Validation' regressor for scikit-learn estimators.

New in mlxtend v0.7.0

**Notes**

The StackingCVRegressor uses scikit-learn's check_cv
internally, which doesn't support a random seed. Thus
NumPy's random seed need to be specified explicitely for
deterministic behavior, for instance, by setting
np.random.seed(RANDOM_SEED)
prior to fitting the StackingCVRegressor

**Parameters**

- `regressors` : array-like, shape = [n_regressors]

    A list of regressors.
    Invoking the `fit` method on the `StackingCVRegressor` will fit clones
    of these original regressors that will
    be stored in the class attribute `self.regr_`.

- `meta_regressor` : object

    The meta-regressor to be fitted on the ensemble of
    regressor

- `cv` : int, cross-validation generator or iterable, optional (default: 5)

    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:
    - None, to use the default 5-fold cross validation,
    - integer, to specify the number of folds in a `KFold`,
    - An object to be used as a cross-validation generator.
    - An iterable yielding train, test splits.
    For integer/None inputs, it will use `KFold` cross-validation

- `use_features_in_secondary` : bool (default: False)

    If True, the meta-regressor will be trained both on
    the predictions of the original regressors and the
    original dataset.
    If False, the meta-regressor will be trained only on
    the predictions of the original regressors.

- `shuffle` : bool (default: True)

    If True,  and the `cv` argument is integer, the training data will
    be shuffled at fitting stage prior to cross-validation. If the `cv`
    argument is a specific cross validation technique, this argument is
    omitted.

- `store_train_meta_features` : bool (default: False)

    If True, the meta-features computed from the training data
    used for fitting the
    meta-regressor stored in the `self.train_meta_features_` array,
    which can be
    accessed after calling `fit`.

- `refit` : bool (default: True)

    Clones the regressors for stacking regression if True (default)
    or else uses the original ones, which will be refitted on the dataset
    upon calling the `fit` method. Setting refit=False is
    recommended if you are working with estimators that are supporting
    the scikit-learn fit/predict API interface but are not compatible
    to scikit-learn's `clone` function.

**Attributes**

- `train_meta_features` : numpy array, shape = [n_samples, n_regressors]

    meta-features for training data, where n_samples is the
    number of samples
    in training data and len(self.regressors) is the number of regressors.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/)

### Methods

<hr>

*fit(X, y, groups=None, sample_weight=None)*

Fit ensemble regressors and the meta-regressor.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : numpy array, shape = [n_samples]

    Target values.


- `groups` : numpy array/None, shape = [n_samples]

    The group that each sample belongs to. This is used by specific
    folding strategies such as GroupKFold()


- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights passed as sample_weights to each regressor
    in the regressors list as well as the meta_regressor.
    Raises error if some regressor does not support
    sample_weight in the fit() method.

**Returns**

- `self` : object


<hr>

*fit_transform(X, y=None, **fit_params)*

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

**Parameters**

- `X` : numpy array of shape [n_samples, n_features]

    Training set.


- `y` : numpy array of shape [n_samples]

    Target values.

**Returns**

- `X_new` : numpy array of shape [n_samples, n_features_new]

    Transformed array.

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

*predict(X)*

Predict target values for X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `y_target` : array-like, shape = [n_samples] or [n_samples, n_targets]

    Predicted target values.

<hr>

*predict_meta_features(X)*

Get meta-features of test-data.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Test vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `meta-features` : numpy array, shape = [n_samples, len(self.regressors)]

    meta-features for test data, where n_samples is the number of
    samples in test data and len(self.regressors) is the number
    of regressors.

<hr>

*score(X, y, sample_weight=None)*

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().

The best possible score is 1.0 and it can be negative (because the

model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

**Parameters**

- `X` : array-like, shape = (n_samples, n_features)

    Test samples. For some estimators this may be a
    precomputed kernel matrix instead, shape = (n_samples,
    n_samples_fitted], where n_samples_fitted is the number of
    samples used in the fitting for the estimator.


- `y` : array-like, shape = (n_samples) or (n_samples, n_outputs)

    True values for X.


- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights.

**Returns**

- `score` : float

    R^2 of self.predict(X) wrt. y.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self




## StackingRegressor

*StackingRegressor(regressors, meta_regressor, verbose=0, use_features_in_secondary=False, store_train_meta_features=False, refit=True)*

A Stacking regressor for scikit-learn estimators for regression.

**Parameters**

- `regressors` : array-like, shape = [n_regressors]

    A list of regressors.
    Invoking the `fit` method on the `StackingRegressor` will fit clones
    of those original regressors that will
    be stored in the class attribute
    `self.regr_`.

- `meta_regressor` : object

    The meta-regressor to be fitted on the ensemble of
    regressors

- `verbose` : int, optional (default=0)

    Controls the verbosity of the building process.
    - `verbose=0` (default): Prints nothing
    - `verbose=1`: Prints the number & name of the regressor being fitted
    - `verbose=2`: Prints info about the parameters of the
    regressor being fitted
    - `verbose>2`: Changes `verbose` param of the underlying regressor to
    self.verbose - 2

- `use_features_in_secondary` : bool (default: False)

    If True, the meta-regressor will be trained both on
    the predictions of the original regressors and the
    original dataset.
    If False, the meta-regressor will be trained only on
    the predictions of the original regressors.

- `store_train_meta_features` : bool (default: False)

    If True, the meta-features computed from the training data
    used for fitting the
    meta-regressor stored in the `self.train_meta_features_` array,
    which can be
    accessed after calling `fit`.


**Attributes**

- `regr_` : list, shape=[n_regressors]

    Fitted regressors (clones of the original regressors)

- `meta_regr_` : estimator

    Fitted meta-regressor (clone of the original meta-estimator)

- `coef_` : array-like, shape = [n_features]

    Model coefficients of the fitted meta-estimator

- `intercept_` : float

    Intercept of the fitted meta-estimator

- `train_meta_features` : numpy array, shape = [n_samples, len(self.regressors)]

    meta-features for training data, where n_samples is the
    number of samples
    in training data and len(self.regressors) is the number of regressors.

- `refit` : bool (default: True)

    Clones the regressors for stacking regression if True (default)
    or else uses the original ones, which will be refitted on the dataset
    upon calling the `fit` method. Setting refit=False is
    recommended if you are working with estimators that are supporting
    the scikit-learn fit/predict API interface but are not compatible
    to scikit-learn's `clone` function.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/)

### Methods

<hr>

*fit(X, y, sample_weight=None)*

Learn weight coefficients from training data for each regressor.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] or [n_samples, n_targets]

    Target values.

- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights passed as sample_weights to each regressor
    in the regressors list as well as the meta_regressor.
    Raises error if some regressor does not support
    sample_weight in the fit() method.

**Returns**

- `self` : object


<hr>

*fit_transform(X, y=None, **fit_params)*

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

**Parameters**

- `X` : numpy array of shape [n_samples, n_features]

    Training set.


- `y` : numpy array of shape [n_samples]

    Target values.

**Returns**

- `X_new` : numpy array of shape [n_samples, n_features_new]

    Transformed array.

<hr>

*get_params(deep=True)*

Return estimator parameter names for GridSearch support.

<hr>

*predict(X)*

Predict target values for X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `y_target` : array-like, shape = [n_samples] or [n_samples, n_targets]

    Predicted target values.

<hr>

*predict_meta_features(X)*

Get meta-features of test-data.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Test vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `meta-features` : numpy array, shape = [n_samples, len(self.regressors)]

    meta-features for test data, where n_samples is the number of
    samples in test data and len(self.regressors) is the number
    of regressors.

<hr>

*score(X, y, sample_weight=None)*

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().

The best possible score is 1.0 and it can be negative (because the

model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

**Parameters**

- `X` : array-like, shape = (n_samples, n_features)

    Test samples. For some estimators this may be a
    precomputed kernel matrix instead, shape = (n_samples,
    n_samples_fitted], where n_samples_fitted is the number of
    samples used in the fitting for the estimator.


- `y` : array-like, shape = (n_samples) or (n_samples, n_outputs)

    True values for X.


- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights.

**Returns**

- `score` : float

    R^2 of self.predict(X) wrt. y.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

### Properties

<hr>

*coef_*

None

<hr>

*intercept_*

None




