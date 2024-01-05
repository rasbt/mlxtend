# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings

import numpy as np
from scipy.optimize import minimize


def create_counterfactual(
    x_reference,
    y_desired,
    model,
    X_dataset,
    y_desired_proba=None,
    lammbda=0.1,
    random_seed=None,
):
    """
    Implementation of the counterfactual method by Wachter et al. 2017

    References:

    - Wachter, S., Mittelstadt, B., & Russell, C. (2017).
    Counterfactual explanations without opening the black box:
     Automated decisions and the GDPR. Harv. JL & Tech., 31, 841.,
     https://arxiv.org/abs/1711.00399

    Parameters
    ----------

    x_reference : array-like, shape=[m_features]
        The data instance (training example) to be explained.

    y_desired : int
        The desired class label for `x_reference`.

    model : estimator
        A (scikit-learn) estimator implementing `.predict()` and/or
        `predict_proba()`.
        - If `model` supports `predict_proba()`, then this is used by
        default for the first loss term,
        `(lambda * model.predict[_proba](x_counterfact) - y_desired[_proba])^2`
        - Otherwise, method will fall back to `predict`.

    X_dataset : array-like, shape=[n_examples, m_features]
        A (training) dataset for picking the initial counterfactual
        as initial value for starting the optimization procedure.

    y_desired_proba : float (default: None)
        A float within the range [0, 1] designating the desired
        class probability for `y_desired`.
        - If `y_desired_proba=None` (default), the first loss term
        is `(lambda * model(x_counterfact) - y_desired)^2` where `y_desired`
        is a class label
        - If `y_desired_proba` is not None, the first loss term
        is `(lambda * model(x_counterfact) - y_desired_proba)^2`

    lammbda : Weighting parameter for the first loss term,
        `(lambda * model(x_counterfact) - y_desired[_proba])^2`

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator for selecting the inital counterfactual
        from `X_dataset`.

    """
    if y_desired_proba is not None:
        use_proba = True
        if not hasattr(model, "predict_proba"):
            raise AttributeError(
                "Your `model` does not support "
                "`predict_proba`. Set `y_desired_proba` "
                " to `None` to use `predict`instead."
            )
    else:
        use_proba = False

    if y_desired_proba is None:
        # class label
        y_to_be_annealed_to = y_desired
    else:
        # class proba corresponding to class label y_desired
        y_to_be_annealed_to = y_desired_proba

    # start with random counterfactual
    rng = np.random.RandomState(random_seed)
    x_counterfact = X_dataset[rng.randint(X_dataset.shape[0])]

    # compute median absolute deviation
    mad = np.abs(np.median(X_dataset, axis=0) - x_reference)

    def dist(x_reference, x_counterfact):
        numerator = np.abs(x_reference - x_counterfact)
        return np.sum(numerator / mad)

    def loss(x_counterfact, lammbda):
        if use_proba:
            y_predict = model.predict_proba(x_counterfact.reshape(1, -1)).flatten()[
                y_desired
            ]
        else:
            y_predict = model.predict(x_counterfact.reshape(1, -1))

        diff = lammbda * (y_predict - y_to_be_annealed_to) ** 2

        return diff + dist(x_reference, x_counterfact)

    res = minimize(loss, x_counterfact, args=(lammbda), method="Nelder-Mead")

    if not res["success"]:
        warnings.warn(res["message"])

    x_counterfact = res["x"]

    return x_counterfact
