# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# License: BSD 3 clause

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone

from mlxtend.classifier import MultiLayerPerceptron, SoftmaxRegression


def _model(estimator, **kwargs):
    params = dict(eta=0.01, epochs=2, minibatches=1, random_seed=7)
    params.update(kwargs)
    if estimator is MultiLayerPerceptron:
        params["hidden_layers"] = [3]
    return estimator(**params)


def _data(n_classes):
    return np.arange(24, dtype=float).reshape(12, 2) / 24, np.arange(12) % n_classes


@pytest.mark.parametrize("estimator", [SoftmaxRegression, MultiLayerPerceptron])
@pytest.mark.parametrize("first,second", [(2, 3), (3, 2)])
def test_refit_infers_the_current_training_class_count(estimator, first, second):
    X, old_y = _data(first)
    _, new_y = _data(second)
    model = _model(estimator).fit(X, old_y)

    model.fit(X, new_y)
    fresh = _model(estimator).fit(X, new_y)

    assert model.predict_proba(X).shape == (len(X), second)
    assert_allclose(model.predict_proba(X), fresh.predict_proba(X), atol=1e-12)
    assert_allclose(model.cost_, fresh.cost_, atol=1e-12)
    assert model.get_params()["n_classes"] is None


@pytest.mark.parametrize("estimator", [SoftmaxRegression, MultiLayerPerceptron])
def test_cloning_fitted_model_preserves_automatic_class_inference(estimator):
    X, y = _data(2)
    _, new_y = _data(3)
    trained = _model(estimator).fit(X, y)

    copied = clone(trained).fit(X, new_y)
    fresh = _model(estimator).fit(X, new_y)

    assert_allclose(copied.predict_proba(X), fresh.predict_proba(X), atol=1e-12)
    assert copied.get_params()["n_classes"] is None


@pytest.mark.parametrize("estimator", [SoftmaxRegression, MultiLayerPerceptron])
def test_explicit_class_count_is_preserved_on_refit(estimator):
    X, y = _data(2)
    _, new_y = _data(3)
    model = _model(estimator, n_classes=4).fit(X, y).fit(X, new_y)
    fresh = _model(estimator, n_classes=4).fit(X, new_y)

    assert model.predict_proba(X).shape == (len(X), 4)
    assert model.get_params()["n_classes"] == 4
    assert_allclose(model.predict_proba(X), fresh.predict_proba(X), atol=1e-12)


@pytest.mark.parametrize("estimator", [SoftmaxRegression, MultiLayerPerceptron])
def test_continue_training_retains_fitted_class_count(estimator):
    X, y = _data(3)
    model = _model(estimator, epochs=1).fit(X, y)
    model.fit(X, y, init_params=False)
    fresh = _model(estimator, epochs=2).fit(X, y)

    assert model.predict_proba(X).shape == (len(X), 3)
    assert_allclose(model.predict_proba(X), fresh.predict_proba(X), atol=1e-12)
    assert_allclose(model.cost_, fresh.cost_, atol=1e-12)
