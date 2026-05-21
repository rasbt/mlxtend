# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Nonparametric Permutation Test
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
import numpy as np


def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(
        sample_indices, size=sample_indices.shape[0], replace=True
    )
    return X[bootstrap_indices], y[bootstrap_indices]


def bias_variance_decomp(
    estimator,
    X_train,
    y_train,
    X_test,
    y_test,
    loss="0-1_loss",
    num_rounds=200,
    random_seed=None,
    **fit_params,
):
    """
    estimator : object
        A classifier or regressor object or class implementing both a
        `fit` and `predict` method similar to the scikit-learn API.

    X_train : array-like or pandas DataFrame, shape=(num_examples, num_features)
    A training dataset for drawing the bootstrap samples to carry
    out the bias-variance decomposition.

    y_train : array-like or pandas Series, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_train` examples.

    X_test : array-like or pandas DataFrame, shape=(num_examples, num_features)
        The test dataset for computing the average loss, bias,
        and variance.

    y_test : array-like or pandas Series, shape=(num_examples)
        Targets (class labels, continuous values in case of regression)
        associated with the `X_test` examples.

    loss : str (default='0-1_loss')
        Loss function for performing the bias-variance decomposition.
        Currently allowed values are '0-1_loss' and 'mse'.

    num_rounds : int (default=200)
        Number of bootstrap rounds (sampling from the training set)
        for performing the bias-variance decomposition. Each bootstrap
        sample has the same size as the original training set.

    random_seed : int (default=None)
        Random seed for the bootstrap sampling used for the
        bias-variance decomposition.

    fit_params : additional parameters
        Additional parameters to be passed to the .fit() function of the
        estimator when it is fit to the bootstrap samples.

    Returns
    ----------
    avg_expected_loss, avg_bias, avg_var : returns the average expected
        loss, average bias, and average variance (all floats), where the
        average is computed over the data points in the test set.

        Note that for the ``'mse'`` loss, ``avg_bias`` reports the
        average **squared** bias
        (``mean((main_predictions - y_test) ** 2)``) — i.e. the term
        that already appears squared in the standard bias-variance
        decomposition ``Loss = Bias^2 + Variance``. For the
        ``'0-1_loss'``, ``avg_bias`` is the misclassification rate of
        the main prediction (the majority vote across bootstrap
        replicates), which is the 0-1 analogue of squared bias.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/bias_variance_decomp/

    """
    supported = ["0-1_loss", "mse"]
    if loss not in supported:
        raise NotImplementedError("loss must be one of the following: %s" % supported)

        # Convert pandas inputs to numpy arrays
    if hasattr(X_train, "loc"):
        X_train = X_train.to_numpy() if hasattr(X_train, "to_numpy") else X_train.values
    if hasattr(y_train, "loc"):
        y_train = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train.values
    if hasattr(X_test, "loc"):
        X_test = X_test.to_numpy() if hasattr(X_test, "to_numpy") else X_test.values
    if hasattr(y_test, "loc"):
        y_test = y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test.values

    rng = np.random.RandomState(random_seed)

    if loss == "0-1_loss":
        dtype = np.int64
    elif loss == "mse":
        dtype = np.float64

    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=dtype)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)

        # Keras support
        if estimator.__class__.__name__ in ["Sequential", "Functional"]:
            # reset model
            for ix, layer in enumerate(estimator.layers):
                if hasattr(estimator.layers[ix], "kernel_initializer") and hasattr(
                    estimator.layers[ix], "bias_initializer"
                ):
                    weight_initializer = estimator.layers[ix].kernel_initializer
                    bias_initializer = estimator.layers[ix].bias_initializer

                    old_weights, old_biases = estimator.layers[ix].get_weights()

                    estimator.layers[ix].set_weights(
                        [
                            weight_initializer(shape=old_weights.shape),
                            bias_initializer(shape=len(old_biases)),
                        ]
                    )

            estimator.fit(X_boot, y_boot, **fit_params)
            pred = estimator.predict(X_test).reshape(1, -1)
        else:
            pred = estimator.fit(X_boot, y_boot, **fit_params).predict(X_test)
        all_pred[i] = pred

    if loss == "0-1_loss":
        main_predictions = np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred
        )

        avg_expected_loss = np.apply_along_axis(
            lambda x: (x != y_test).mean(), axis=1, arr=all_pred
        ).mean()

        avg_bias = np.sum(main_predictions != y_test) / y_test.size

        var = np.zeros(pred.shape)

        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int_)
        var /= num_rounds

        avg_var = var.sum() / y_test.shape[0]

    else:
        avg_expected_loss = np.apply_along_axis(
            lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred
        ).mean()

        main_predictions = np.mean(all_pred, axis=0)

        # `avg_bias` here is the average *squared* bias — the Bias^2
        # term in the standard decomposition Loss = Bias^2 + Variance.
        # See the Returns section of the docstring; the variable is
        # kept named `avg_bias` for backwards compatibility.
        avg_bias = np.sum((main_predictions - y_test) ** 2) / y_test.size
        avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size

    return avg_expected_loss, avg_bias, avg_var
