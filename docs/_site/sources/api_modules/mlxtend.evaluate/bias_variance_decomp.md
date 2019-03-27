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

