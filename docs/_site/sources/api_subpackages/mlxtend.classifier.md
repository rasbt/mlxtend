mlxtend version: 0.15.0dev 
## Adaline

*Adaline(eta=0.01, epochs=50, minibatches=None, random_seed=None, print_progress=0)*

ADAptive LInear NEuron classifier.

Note that this implementation of Adaline expects binary class labels
in {0, 1}.

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
    If len(y): Stochastic Gradient Descent (SGD) online learning
    If 1 < minibatches < len(y): SGD Minibatch learning

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

    Sum of squared errors after each epoch.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/](http://rasbt.github.io/mlxtend/user_guide/classifier/Adaline/)

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

*score(X, y)*

Compute the prediction accuracy

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values (true class labels).

**Returns**

- `acc` : float

    The prediction accuracy as a float
    between 0.0 and 1.0 (perfect score).

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




## EnsembleVoteClassifier

*EnsembleVoteClassifier(clfs, voting='hard', weights=None, verbose=0, refit=True)*

Soft Voting/Majority Rule classifier for scikit-learn estimators.

**Parameters**

- `clfs` : array-like, shape = [n_classifiers]

    A list of classifiers.
    Invoking the `fit` method on the `VotingClassifier` will fit clones
    of those original classifiers that will
    be stored in the class attribute
    `self.clfs_` if `refit=True` (default).

- `voting` : str, {'hard', 'soft'} (default='hard')

    If 'hard', uses predicted class labels for majority rule voting.
    Else if 'soft', predicts the class label based on the argmax of
    the sums of the predicted probalities, which is recommended for
    an ensemble of well-calibrated classifiers.

- `weights` : array-like, shape = [n_classifiers], optional (default=`None`)

    Sequence of weights (`float` or `int`) to weight the occurances of
    predicted class labels (`hard` voting) or class probabilities
    before averaging (`soft` voting). Uses uniform weights if `None`.

- `verbose` : int, optional (default=0)

    Controls the verbosity of the building process.
    - `verbose=0` (default): Prints nothing
    - `verbose=1`: Prints the number & name of the clf being fitted
    - `verbose=2`: Prints info about the parameters of the clf being fitted
    - `verbose>2`: Changes `verbose` param of the underlying clf to
    self.verbose - 2

- `refit` : bool (default: True)

    Refits classifiers in `clfs` if True; uses references to the `clfs`,
    otherwise (assumes that the classifiers were already fit).
    Note: refit=False is incompatible to mist scikit-learn wrappers!
    For instance, if any form of cross-validation is performed
    this would require the re-fitting classifiers to training folds, which
    would raise a NotFitterError if refit=False.
    (New in mlxtend v0.6.)

**Attributes**

- `classes_` : array-like, shape = [n_predictions]


- `clf` : array-like, shape = [n_predictions]

    The unmodified input classifiers

- `clf_` : array-like, shape = [n_predictions]

    Fitted clones of the input classifiers

**Examples**

    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from mlxtend.sklearn import EnsembleVoteClassifier
    >>> clf1 = LogisticRegression(random_seed=1)
    >>> clf2 = RandomForestClassifier(random_seed=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ... voting='hard', verbose=1)
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/](http://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/)

### Methods

<hr>

*fit(X, y, sample_weight=None)*

Learn weight coefficients from training data for each classifier.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

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

Predict class labels for X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `maj` : array-like, shape = [n_samples]

    Predicted class labels.

<hr>

*predict_proba(X)*

Predict class probabilities for X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `avg` : array-like, shape = [n_samples, n_classes]

    Weighted average probability for each class per sample.

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

<hr>

*transform(X)*

Return class labels or probabilities for X for each estimator.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `If `voting='soft'`` : array-like = [n_classifiers, n_samples, n_classes]

    Class probabilties calculated by each classifier.

- `If `voting='hard'`` : array-like = [n_classifiers, n_samples]

    Class labels predicted by each classifier.




## LogisticRegression

*LogisticRegression(eta=0.01, epochs=50, l2_lambda=0.0, minibatches=1, random_seed=None, print_progress=0)*

Logistic regression classifier.

Note that this implementation of Logistic Regression
expects binary class labels in {0, 1}.

**Parameters**

- `eta` : float (default: 0.01)

    Learning rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Passes over the training dataset.
    Prior to each epoch, the dataset is shuffled
    if `minibatches > 1` to prevent cycles in stochastic gradient descent.

- `l2_lambda` : float

    Regularization parameter for L2 regularization.
    No regularization if l2_lambda=0.0.

- `minibatches` : int (default: 1)

    The number of minibatches for gradient-based optimization.
    If 1: Gradient Descent learning
    If len(y): Stochastic Gradient Descent (SGD) online learning
    If 1 < minibatches < len(y): SGD Minibatch learning

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
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

    List of floats with cross_entropy cost (sgd or gd) for every
    epoch.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/](http://rasbt.github.io/mlxtend/user_guide/classifier/LogisticRegression/)

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

*predict_proba(X)*

Predict class probabilities of X from the net input.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `Class 1 probability` : float


<hr>

*score(X, y)*

Compute the prediction accuracy

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values (true class labels).

**Returns**

- `acc` : float

    The prediction accuracy as a float
    between 0.0 and 1.0 (perfect score).

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




## MultiLayerPerceptron

*MultiLayerPerceptron(eta=0.5, epochs=50, hidden_layers=[50], n_classes=None, momentum=0.0, l1=0.0, l2=0.0, dropout=1.0, decrease_const=0.0, minibatches=1, random_seed=None, print_progress=0)*

Multi-layer perceptron classifier with logistic sigmoid activations

**Parameters**

- `eta` : float (default: 0.5)

    Learning rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Passes over the training dataset.
    Prior to each epoch, the dataset is shuffled
    if `minibatches > 1` to prevent cycles in stochastic gradient descent.

- `hidden_layers` : list (default: [50])

    Number of units per hidden layer. By default 50 units in the
    first hidden layer. At the moment only 1 hidden layer is supported

- `n_classes` : int (default: None)

    A positive integer to declare the number of class labels
    if not all class labels are present in a partial training set.
    Gets the number of class labels automatically if None.

- `l1` : float (default: 0.0)

    L1 regularization strength

- `l2` : float (default: 0.0)

    L2 regularization strength

- `momentum` : float (default: 0.0)

    Momentum constant. Factor multiplied with the
    gradient of the previous epoch t-1 to improve
    learning speed
    w(t) := w(t) - (grad(t) + momentum * grad(t-1))

- `decrease_const` : float (default: 0.0)

    Decrease constant. Shrinks the learning rate
    after each epoch via eta / (1 + epoch*decrease_const)

- `minibatches` : int (default: 1)

    Divide the training data into *k* minibatches
    for accelerated stochastic gradient descent learning.
    Gradient Descent Learning if `minibatches` = 1
    Stochastic Gradient Descent learning if `minibatches` = len(y)
    Minibatch learning if `minibatches` > 1

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
    0: No output
    1: Epochs elapsed and cost
    2: 1 plus time elapsed
    3: 2 plus estimated time until completion

**Attributes**

- `w_` : 2d-array, shape=[n_features, n_classes]

    Weights after fitting.

- `b_` : 1D-array, shape=[n_classes]

    Bias units after fitting.

- `cost_` : list

    List of floats; the mean categorical cross entropy
    cost after each epoch.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/MultiLayerPerceptron/](http://rasbt.github.io/mlxtend/user_guide/classifier/MultiLayerPerceptron/)

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

*predict_proba(X)*

Predict class probabilities of X from the net input.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `Class probabilties` : array-like, shape= [n_samples, n_classes]


<hr>

*score(X, y)*

Compute the prediction accuracy

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values (true class labels).

**Returns**

- `acc` : float

    The prediction accuracy as a float
    between 0.0 and 1.0 (perfect score).

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




## Perceptron

*Perceptron(eta=0.1, epochs=50, random_seed=None, print_progress=0)*

Perceptron classifier.

Note that this implementation of the Perceptron expects binary class labels
in {0, 1}.

**Parameters**

- `eta` : float (default: 0.1)

    Learning rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Number of passes over the training dataset.
    Prior to each epoch, the dataset is shuffled to prevent cycles.

- `random_seed` : int

    Random state for initializing random weights and shuffling.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
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

    Number of misclassifications in every epoch.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/Perceptron/](http://rasbt.github.io/mlxtend/user_guide/classifier/Perceptron/)

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

*score(X, y)*

Compute the prediction accuracy

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values (true class labels).

**Returns**

- `acc` : float

    The prediction accuracy as a float
    between 0.0 and 1.0 (perfect score).

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




## SoftmaxRegression

*SoftmaxRegression(eta=0.01, epochs=50, l2=0.0, minibatches=1, n_classes=None, random_seed=None, print_progress=0)*

Softmax regression classifier.

**Parameters**

- `eta` : float (default: 0.01)

    Learning rate (between 0.0 and 1.0)

- `epochs` : int (default: 50)

    Passes over the training dataset.
    Prior to each epoch, the dataset is shuffled
    if `minibatches > 1` to prevent cycles in stochastic gradient descent.

- `l2` : float

    Regularization parameter for L2 regularization.
    No regularization if l2=0.0.

- `minibatches` : int (default: 1)

    The number of minibatches for gradient-based optimization.
    If 1: Gradient Descent learning
    If len(y): Stochastic Gradient Descent (SGD) online learning
    If 1 < minibatches < len(y): SGD Minibatch learning

- `n_classes` : int (default: None)

    A positive integer to declare the number of class labels
    if not all class labels are present in a partial training set.
    Gets the number of class labels automatically if None.

- `random_seed` : int (default: None)

    Set random state for shuffling and initializing the weights.

- `print_progress` : int (default: 0)

    Prints progress in fitting to stderr.
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

    List of floats, the average cross_entropy for each epoch.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/](http://rasbt.github.io/mlxtend/user_guide/classifier/SoftmaxRegression/)

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

*predict_proba(X)*

Predict class probabilities of X from the net input.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `Class probabilties` : array-like, shape= [n_samples, n_classes]


<hr>

*score(X, y)*

Compute the prediction accuracy

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values (true class labels).

**Returns**

- `acc` : float

    The prediction accuracy as a float
    between 0.0 and 1.0 (perfect score).

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




## StackingClassifier

*StackingClassifier(classifiers, meta_classifier, use_probas=False, average_probas=False, verbose=0, use_features_in_secondary=False, store_train_meta_features=False, use_clones=True)*

A Stacking classifier for scikit-learn estimators for classification.

**Parameters**

- `classifiers` : array-like, shape = [n_classifiers]

    A list of classifiers.
    Invoking the `fit` method on the `StackingClassifer` will fit clones
    of these original classifiers that will
    be stored in the class attribute
    `self.clfs_`.

- `meta_classifier` : object

    The meta-classifier to be fitted on the ensemble of
    classifiers

- `use_probas` : bool (default: False)

    If True, trains meta-classifier based on predicted probabilities
    instead of class labels.

- `average_probas` : bool (default: False)

    Averages the probabilities as meta features if True.

- `verbose` : int, optional (default=0)

    Controls the verbosity of the building process.
    - `verbose=0` (default): Prints nothing
    - `verbose=1`: Prints the number & name of the regressor being fitted
    - `verbose=2`: Prints info about the parameters of the
    regressor being fitted
    - `verbose>2`: Changes `verbose` param of the underlying regressor to
    self.verbose - 2

- `use_features_in_secondary` : bool (default: False)

    If True, the meta-classifier will be trained both on the predictions
    of the original classifiers and the original dataset.
    If False, the meta-classifier will be trained only on the predictions
    of the original classifiers.

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
    StackingClassifier's `fit` method.
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
    [http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/](http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/)

### Methods

<hr>

*fit(X, y, sample_weight=None)*

Fit ensemble classifers and the meta-classifier.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] or [n_samples, n_outputs]

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

- `labels` : array-like, shape = [n_samples] or [n_samples, n_outputs]

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

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `proba` : array-like, shape = [n_samples, n_classes] or a list of                 n_outputs of such arrays if n_outputs > 1.

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




