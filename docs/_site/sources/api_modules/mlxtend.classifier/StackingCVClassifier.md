## StackingCVClassifier

*StackingCVClassifier(classifiers, meta_classifier, use_probas=False, cv=2, use_features_in_secondary=False, stratify=True, shuffle=True, verbose=0, store_train_meta_features=False, use_clones=True)*

A 'Stacking Cross-Validation' classifier for scikit-learn estimators.

New in mlxtend v0.4.3

**Notes**

The StackingCVClassifier uses scikit-learn's check_cv
internally, which doesn't support a random seed. Thus
NumPy's random seed need to be specified explicitely for
deterministic behavior, for instance, by setting
np.random.seed(RANDOM_SEED)
prior to fitting the StackingCVClassifier

**Parameters**

- `classifiers` : array-like, shape = [n_classifiers]

    A list of classifiers.
    Invoking the `fit` method on the `StackingCVClassifer` will fit clones
    of these original classifiers that will
    be stored in the class attribute `self.clfs_`.

- `meta_classifier` : object

    The meta-classifier to be fitted on the ensemble of
    classifiers

- `use_probas` : bool (default: False)

    If True, trains meta-classifier based on predicted probabilities
    instead of class labels.

- `cv` : int, cross-validation generator or an iterable, optional (default: 2)

    Determines the cross-validation splitting strategy.
    Possible inputs for cv are:
    - None, to use the default 2-fold cross validation,
    - integer, to specify the number of folds in a `(Stratified)KFold`,
    - An object to be used as a cross-validation generator.
    - An iterable yielding train, test splits.
    For integer/None inputs, it will use either a `KFold` or
    `StratifiedKFold` cross validation depending the value of `stratify`
    argument.

- `use_features_in_secondary` : bool (default: False)

    If True, the meta-classifier will be trained both on the predictions
    of the original classifiers and the original dataset.
    If False, the meta-classifier will be trained only on the predictions
    of the original classifiers.

- `stratify` : bool (default: True)

    If True, and the `cv` argument is integer it will follow a stratified
    K-Fold cross validation technique. If the `cv` argument is a specific
    cross validation technique, this argument is omitted.

- `shuffle` : bool (default: True)

    If True,  and the `cv` argument is integer, the training data will be
    shuffled at fitting stage prior to cross-validation. If the `cv`
    argument is a specific cross validation technique, this argument is
    omitted.

- `verbose` : int, optional (default=0)

    Controls the verbosity of the building process.
    - `verbose=0` (default): Prints nothing
    - `verbose=1`: Prints the number & name of the regressor being fitted
    and which fold is currently being used for fitting
    - `verbose=2`: Prints info about the parameters of the
    regressor being fitted
    - `verbose>2`: Changes `verbose` param of the underlying regressor to
    self.verbose - 2

- `store_train_meta_features` : bool (default: False)

    If True, the meta-features computed from the training data used
    for fitting the meta-classifier stored in the
    `self.train_meta_features_` array, which can be
    accessed after calling `fit`.

- `use_clones` : bool (default: True)

    Clones the classifiers for stacking classification if True (default)
    or else uses the original ones, which will be refitted on the dataset
    upon calling the `fit` method. Hence, if use_clones=True, the original
    input classifiers will remain unmodified upon using the
    StackingCVClassifier's `fit` method.
    Setting `use_clones=False` is
    recommended if you are working with estimators that are supporting
    the scikit-learn fit/predict API interface but are not compatible
    to scikit-learn's `clone` function.


**Attributes**

- `clfs_` : list, shape=[n_classifiers]

    Fitted classifiers (clones of the original classifiers)

- `meta_clf_` : estimator

    Fitted meta-classifier (clone of the original meta-estimator)

- `train_meta_features` : numpy array, shape = [n_samples, n_classifiers]

    meta-features for training data, where n_samples is the
    number of samples
    in training data and n_classifiers is the number of classfiers.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingCVClassifier/)

### Methods

<hr>

*fit(X, y, groups=None, sample_weight=None)*

Fit ensemble classifers and the meta-classifier.

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

Return estimator parameter names for GridSearch support.

<hr>

*predict(X)*

Predict target values for X.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `labels` : array-like, shape = [n_samples]

    Predicted class labels.

<hr>

*predict_meta_features(X)*

Get meta-features of test-data.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Test vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `meta-features` : numpy array, shape = [n_samples, n_classifiers]

    Returns the meta-features for test data.

<hr>

*predict_proba(X)*

Predict class probabilities for X.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `proba` : array-like, shape = [n_samples, n_classes]

    Probability for each class per sample.

<hr>

*score(X, y, sample_weight=None)*

Returns the mean accuracy on the given test data and labels.

In multi-label classification, this is the subset accuracy
which is a harsh metric since you require for each sample that
each label set be correctly predicted.

**Parameters**

- `X` : array-like, shape = (n_samples, n_features)

    Test samples.


- `y` : array-like, shape = (n_samples) or (n_samples, n_outputs)

    True labels for X.


- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights.

**Returns**

- `score` : float

    Mean accuracy of self.predict(X) wrt. y.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

