## PrincipalComponentAnalysis

*PrincipalComponentAnalysis(n_components=None, solver='svd', whitening=False)*

Principal Component Analysis Class

**Parameters**

- `n_components` : int (default: None)

    The number of principal components for transformation.
    Keeps the original dimensions of the dataset if `None`.

- `solver` : str (default: 'svd')

    Method for performing the matrix decomposition.
    {'eigen', 'svd'}

- `whitening` : bool (default: False)

    Performs whitening such that the covariance matrix of
    the transformed data will be the identity matrix.

**Attributes**

- `w_` : array-like, shape=[n_features, n_components]

    Projection matrix

- `e_vals_` : array-like, shape=[n_features]

    Eigenvalues in sorted order.

- `e_vecs_` : array-like, shape=[n_features]

    Eigenvectors in sorted order.

- `loadings_` : array_like, shape=[n_features, n_features]

    The factor loadings of the original variables onto
    the principal components. The columns are the principal
    components, and the rows are the features loadings.
    For instance, the first column contains the loadings onto
    the first principal component. Note that the signs may
    be flipped depending on whether you use the 'eigen' or
    'svd' solver; this does not affect the interpretation
    of the loadings though.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_extraction/PrincipalComponentAnalysis/](http://rasbt.github.io/mlxtend/user_guide/feature_extraction/PrincipalComponentAnalysis/)

### Methods

<hr>

*fit(X, y=None)*

Learn model from training data.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

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

- `X_projected` : np.ndarray, shape = [n_samples, n_components]

    Projected training vectors.

