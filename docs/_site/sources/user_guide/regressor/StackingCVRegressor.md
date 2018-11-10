# StackingCVRegressor

An ensemble-learning meta-regressor for stacking regression

> from mlxtend.regressor import StackingCVRegressor

## Overview

Stacking is an ensemble learning technique to combine multiple regression models via a meta-regressor. The `StackingCVRegressor` extends the standard stacking algorithm (implemented as [`StackingRegressor`](StackingRegressor.md)) using out-of-fold predictions to prepare the input data for the level-2 regressor.

In the standard stacking procedure, the first-level regressors are fit to the same training set that is used prepare the inputs for the second-level regressor, which may lead to overfitting. The `StackingCVRegressor`, however, uses the concept of out-of-fold predictions: the dataset is split into k folds, and in k successive rounds, k-1 folds are used to fit the first level regressor. In each round, the first-level regressors are then applied to the remaining 1 subset that was not used for model fitting in each iteration. The resulting predictions are then stacked and provided -- as input data -- to the second-level regressor. After the training of the `StackingCVRegressor`, the first-level regressors are fit to the entire dataset for optimal predicitons.

![](./StackingCVRegressor_files/stacking_cv_regressor_overview.png)

### References


- Breiman, Leo. "[Stacked regressions.](http://link.springer.com/article/10.1023/A:1018046112532#page-1)" Machine learning 24.1 (1996): 49-64.
- Analogous implementation: [`StackingCVClassifier`](../classifier/StackingCVClassifier.md)

## Example 1: Boston Housing Data Predictions

In this example we evaluate some basic prediction models on the boston housing dataset and see how the $R^2$ and MSE scores are affected by combining the models with `StackingCVRegressor`. The code output below demonstrates that the stacked model performs the best on this dataset -- slightly better than the best single regression model.


```python
from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import load_boston
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import numpy as np

RANDOM_SEED = 42

X, y = load_boston(return_X_y=True)

svr = SVR(kernel='linear')
lasso = Lasso()
rf = RandomForestRegressor(n_estimators=5, 
                           random_state=RANDOM_SEED)

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors=(svr, lasso, rf),
                            meta_regressor=lasso)

print('5-fold cross validation scores:\n')

for clf, label in zip([svr, lasso, rf, stack], ['SVM', 'Lasso', 
                                                'Random Forest', 
                                                'StackingCVRegressor']):
    scores = cross_val_score(clf, X, y, cv=5)
    print("R^2 Score: %0.2f (+/- %0.2f) [%s]" % (
        scores.mean(), scores.std(), label))
```

    5-fold cross validation scores:
    
    R^2 Score: 0.45 (+/- 0.29) [SVM]
    R^2 Score: 0.43 (+/- 0.14) [Lasso]
    R^2 Score: 0.52 (+/- 0.28) [Random Forest]
    R^2 Score: 0.58 (+/- 0.24) [StackingCVRegressor]



```python
# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors=(svr, lasso, rf),
                            meta_regressor=lasso)

print('5-fold cross validation scores:\n')

for clf, label in zip([svr, lasso, rf, stack], ['SVM', 'Lasso', 
                                                'Random Forest', 
                                                'StackingCVRegressor']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='neg_mean_squared_error')
    print("Neg. MSE Score: %0.2f (+/- %0.2f) [%s]" % (
        scores.mean(), scores.std(), label))
```

    5-fold cross validation scores:
    
    Neg. MSE Score: -33.69 (+/- 22.36) [SVM]
    Neg. MSE Score: -35.53 (+/- 16.99) [Lasso]
    Neg. MSE Score: -27.32 (+/- 16.62) [Random Forest]
    Neg. MSE Score: -25.64 (+/- 18.11) [StackingCVRegressor]


## Example 2: GridSearchCV with Stacking

In this second example we demonstrate how `StackingCVRegressor` works in combination with `GridSearchCV`. The stack still allows tuning hyper parameters of the base and meta models!

To set up a parameter grid for scikit-learn's `GridSearch`, we simply provide the estimator's names in the parameter grid -- in the special case of the meta-regressor, we append the `'meta-'` prefix.



```python
from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import load_boston
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

X, y = load_boston(return_X_y=True)

ridge = Ridge()
lasso = Lasso()
rf = RandomForestRegressor(random_state=RANDOM_SEED)

# The StackingCVRegressor uses scikit-learn's check_cv
# internally, which doesn't support a random seed. Thus
# NumPy's random seed need to be specified explicitely for
# deterministic behavior
np.random.seed(RANDOM_SEED)
stack = StackingCVRegressor(regressors=(lasso, ridge),
                            meta_regressor=rf, 
                            use_features_in_secondary=True)

params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0]}

grid = GridSearchCV(
    estimator=stack, 
    param_grid={
        'lasso__alpha': [x/5.0 for x in range(1, 10)],
        'ridge__alpha': [x/20.0 for x in range(1, 10)],
        'meta-randomforestregressor__n_estimators': [10, 100]
    }, 
    cv=5,
    refit=True
)

grid.fit(X, y)

print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
```

    Best: 0.673590 using {'lasso__alpha': 0.4, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.3}



```python
cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))
    if r > 10:
        break
print('...')

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)
```

    0.622 +/- 0.10 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.05}
    0.649 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.1}
    0.650 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.15}
    0.667 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.2}
    0.629 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.25}
    0.663 +/- 0.08 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.3}
    0.633 +/- 0.08 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.35}
    0.637 +/- 0.08 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.4}
    0.649 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.45}
    0.653 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 100, 'ridge__alpha': 0.05}
    0.648 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 100, 'ridge__alpha': 0.1}
    0.645 +/- 0.09 {'lasso__alpha': 0.2, 'meta-randomforestregressor__n_estimators': 100, 'ridge__alpha': 0.15}
    ...
    Best parameters: {'lasso__alpha': 0.4, 'meta-randomforestregressor__n_estimators': 10, 'ridge__alpha': 0.3}
    Accuracy: 0.67


**Note**

The `StackingCVRegressor` also enables grid search over the `regressors` argument. However, due to the current implementation of `GridSearchCV` in scikit-learn, it is not possible to search over both, different regressors and regressor parameters at the same time. For instance, while the following parameter dictionary works

    params = {'randomforestregressor__n_estimators': [1, 100],
    'regressors': [(regr1, regr1, regr1), (regr2, regr3)]}
    
it will use the instance settings of `regr1`, `regr2`, and `regr3` and not overwrite it with the `'n_estimators'` settings from `'randomforestregressor__n_estimators': [1, 100]`.

## API


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

    Test samples.


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


