# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlxtend.classifier import MultiLayerPerceptron


def _model(l1, l2=0, epochs=1):
    return MultiLayerPerceptron(
        hidden_layers=[3],
        n_classes=2,
        eta=0.01,
        epochs=epochs,
        l1=l1,
        l2=l2,
        random_seed=5,
        minibatches=1,
    )


def _data():
    return np.arange(18, dtype=float).reshape(6, 3) / 10, np.array([0, 1, 0, 1, 0, 1])


@pytest.mark.parametrize("l1", [0.0, 0.1, 0.8])
@pytest.mark.parametrize("l2", [0.0, 0.2])
def test_mlp_l1_update_matches_the_reported_penalty_gradient(l1, l2):
    X, y = _data()
    penalty_model = _model(l1, epochs=0).fit(X, y)
    unpenalized = _model(0, l2=l2).fit(X, y)
    penalized = _model(l1, l2=l2).fit(X, y)
    step = 1e-7

    for layer in ["1", "out"]:
        weights = penalty_model.w_[layer]
        numerical_gradient = np.empty_like(weights)
        for index in np.ndindex(weights.shape):
            original = weights[index]
            weights[index] = original + step
            upper = penalty_model._compute_cost(np.zeros(len(X)))
            weights[index] = original - step
            lower = penalty_model._compute_cost(np.zeros(len(X)))
            weights[index] = original
            numerical_gradient[index] = (upper - lower) / (2 * step)

        observed_gradient = (
            unpenalized.w_[layer] - penalized.w_[layer]
        ) / penalized.eta
        assert_allclose(observed_gradient, numerical_gradient, rtol=1e-6, atol=1e-9)
        assert_allclose(penalized.b_[layer], unpenalized.b_[layer], rtol=0, atol=0)


def test_mlp_l1_uses_a_zero_subgradient_at_zero_weights():
    X, y = _data()
    baseline = _model(0, epochs=0).fit(X, y)
    penalized = _model(0.8, epochs=0).fit(X, y)
    for model in [baseline, penalized]:
        for layer in ["1", "out"]:
            model.w_[layer][0, 0] = 0
        model.epochs = 1
        model.fit(X, y, init_params=False)

    for layer in ["1", "out"]:
        assert_allclose(penalized.w_[layer][0, 0], baseline.w_[layer][0, 0], atol=1e-12)
