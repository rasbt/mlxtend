# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from mlxtend.classifier import MultiLayerPerceptron


def _model(epochs, minibatches, decrease_const, eta=0.05):
    return MultiLayerPerceptron(
        hidden_layers=[3],
        n_classes=2,
        eta=eta,
        epochs=epochs,
        minibatches=minibatches,
        decrease_const=decrease_const,
        random_seed=4,
    )


def _data():
    # Identical rows isolate the schedule from shuffled minibatch ordering.
    return np.tile([[0.2, -0.4, 0.7]], (6, 1)), np.zeros(6, dtype=int)


def _assert_same_parameters(actual, expected):
    for layer in ["1", "out"]:
        assert_allclose(actual.w_[layer], expected.w_[layer], rtol=1e-12, atol=1e-12)
        assert_allclose(actual.b_[layer], expected.b_[layer], rtol=1e-12, atol=1e-12)
    assert_allclose(actual.cost_, expected.cost_, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("minibatches", [1, 2, 3])
@pytest.mark.parametrize("decay", [0.1, 0.5])
def test_mlp_decay_matches_explicit_per_epoch_learning_rates(minibatches, decay):
    X, y = _data()
    model = _model(epochs=4, minibatches=minibatches, decrease_const=decay).fit(X, y)
    reference = _model(epochs=1, minibatches=minibatches, decrease_const=0)
    for epoch in range(4):
        reference.eta = 0.05 / (1 + epoch * decay)
        # Supply a constant rate to the existing one-epoch update loop.
        reference._decr_eta = reference.eta
        reference.fit(X, y, init_params=epoch == 0)

    _assert_same_parameters(model, reference)
    assert model._decr_eta == pytest.approx(0.05 / (1 + 3 * decay))


def test_mlp_decay_continues_across_fit_calls():
    X, y = _data()
    continued = _model(epochs=2, minibatches=2, decrease_const=0.5).fit(X, y)

    continued.fit(X, y, init_params=False)
    uninterrupted = _model(epochs=4, minibatches=2, decrease_const=0.5).fit(X, y)

    _assert_same_parameters(continued, uninterrupted)


def test_mlp_reinitialization_restarts_the_decay_schedule():
    X, y = _data()
    model = _model(epochs=3, minibatches=2, decrease_const=0.5).fit(X, y)

    model.fit(X, y, init_params=True)
    fresh = _model(epochs=3, minibatches=2, decrease_const=0.5).fit(X, y)

    _assert_same_parameters(model, fresh)


def test_mlp_without_decay_keeps_a_constant_learning_rate():
    X, y = _data()
    model = _model(epochs=3, minibatches=2, decrease_const=0).fit(X, y)

    assert model._decr_eta == model.eta
