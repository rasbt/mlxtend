## LinearDiscriminantAnalysis

*LinearDiscriminantAnalysis(n_discriminants=None)*

Linear Discriminant Analysis Class

**Parameters**

- `n_discriminants` : int (default: None)

    The number of discrimants for transformation.
    Keeps the original dimensions of the dataset if `None`.

**Attributes**

- `w_` : array-like, shape=[n_features, n_discriminants]

    Projection matrix

- `e_vals_` : array-like, shape=[n_features]

    Eigenvalues in sorted order.

- `e_vecs_` : array-like, shape=[n_features]

    Eigenvectors in sorted order.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/](http://rasbt.github.io/mlxtend/user_guide/feature_extraction/LinearDiscriminantAnalysis/)

### Methods

<hr>

*fit(X, y, n_classes=None)*

Fit the LDA model with X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples]

    Target values.

- `n_classes` : int (default: None)

    A positive integer to declare the number of class labels
    if not all class labels are present in a partial training set.
    Gets the number of class labels automatically if None.

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

<hr>

*transform(X)*

Apply the linear transformation on X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `X_projected` : np.ndarray, shape = [n_samples, n_discriminants]

    Projected training vectors.

