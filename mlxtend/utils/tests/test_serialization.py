import os

import numpy as np
import pytest

from mlxtend.classifier import Perceptron
from mlxtend.utils.serialization import load_model_from_json, save_model_to_json


def test_serialization_perceptron():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])

    ppn = Perceptron(epochs=5, eta=0.1)
    ppn.fit(X, y)

    filename = "temp_ppn_model.json"

    try:
        save_model_to_json(ppn, filename)
        assert os.path.exists(filename)

        ppn_loaded = load_model_from_json(filename)

        assert ppn.__class__ == ppn_loaded.__class__
        np.testing.assert_array_almost_equal(ppn.w_, ppn_loaded.w_)
        orig_pred = ppn.predict(X)
        load_pred = ppn_loaded.predict(X)
        np.testing.assert_array_equal(orig_pred, load_pred)
    finally:
        if os.path.exists(filename):
            os.remove(filename)


def test_encoder_fallback():
    import json

    from mlxtend.utils.serialization import MlxtendEncoder

    data = {"complex_obj": open}
    encoded = json.dumps(data, cls=MlxtendEncoder)
    assert "built-in function open" in encoded
