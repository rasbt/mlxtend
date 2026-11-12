# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlxtend.classifier import MultiLayerPerceptron


def _model(momentum, epochs):
    return MultiLayerPerceptron(
        hidden_layers=[3],
        n_classes=2,
        eta=0.01,
        momentum=momentum,
        epochs=epochs,
        minibatches=1,
        random_seed=4,
    )


@pytest.mark.parametrize("momentum", [0.0, 0.1, 0.6])
@pytest.mark.parametrize("restore", [False, True])
def test_mlp_momentum_continuation_matches_uninterrupted_training(momentum, restore):
    X = np.arange(18, dtype=float).reshape(6, 3) / 10 - 0.8
    y = np.array([0, 1, 0, 1, 0, 1])
    continued = _model(momentum, epochs=1).fit(X, y)
    if restore:
        continued = pickle.loads(pickle.dumps(continued))

    continued.fit(X, y, init_params=False)
    uninterrupted = _model(momentum, epochs=2).fit(X, y)

    for layer in ["1", "out"]:
        assert_allclose(continued.w_[layer], uninterrupted.w_[layer], atol=1e-12)
        assert_allclose(continued.b_[layer], uninterrupted.b_[layer], atol=1e-12)
    assert_allclose(continued.cost_, uninterrupted.cost_, atol=1e-12)


def test_mlp_reinitialization_resets_momentum_history():
    X = np.arange(18, dtype=float).reshape(6, 3) / 10 - 0.8
    y = np.array([0, 1, 0, 1, 0, 1])
    model = _model(momentum=0.3, epochs=2).fit(X, y)

    model.fit(X, y, init_params=True)
    fresh = _model(momentum=0.3, epochs=2).fit(X, y)

    for layer in ["1", "out"]:
        assert_allclose(model.w_[layer], fresh.w_[layer], atol=1e-12)
        assert_allclose(model.b_[layer], fresh.b_[layer], atol=1e-12)
    assert_allclose(model.cost_, fresh.cost_, atol=1e-12)
