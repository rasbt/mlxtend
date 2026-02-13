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
    feature_names_to_vary=None,
):
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
        y_to_be_annealed_to = y_desired
    else:
        y_to_be_annealed_to = y_desired_proba
    rng = np.random.RandomState(random_seed)
    x_counterfact = X_dataset[rng.randint(X_dataset.shape[0])]

    mad = np.abs(np.median(X_dataset, axis=0) - x_reference)
    mad[mad == 0] = 1.0

    def dist(x_ref, x_cf):
        numerator = np.abs(x_ref - x_cf)
        return np.sum(numerator / mad)

    def loss(x_curr, lammbda):
        if feature_names_to_vary is not None:
            x_full = np.copy(x_reference)
            x_full[feature_names_to_vary] = x_curr
        else:
            x_full = x_curr
        if use_proba:
            y_predict = model.predict_proba(x_full.reshape(1, -1)).flatten()[y_desired]
        else:
            y_predict = model.predict(x_full.reshape(1, -1))
        diff = lammbda * (y_predict - y_to_be_annealed_to) ** 2
        return diff + dist(x_reference, x_full)

    if feature_names_to_vary is not None:
        initial_guess = x_counterfact[feature_names_to_vary]
    else:
        initial_guess = x_counterfact
    res = minimize(loss, initial_guess, args=(lammbda), method="Nelder-Mead")

    if not res["success"]:
        warnings.warn(res["message"])
    if feature_names_to_vary is not None:
        final_cf = np.copy(x_reference)
        final_cf[feature_names_to_vary] = res["x"]
    else:
        final_cf = res["x"]
    return final_cf
