import importlib
import json

import numpy as np


class MlxtendEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and fallback to strings."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        try:
            return super(MlxtendEncoder, self).default(obj)
        except TypeError:
            return str(obj)


def save_model_to_json(model, filename):
    """Save an mlxtend estimator to a JSON file."""
    model_data = model.__dict__.copy()
    model_data["__module__"] = model.__class__.__module__
    model_data["__class__"] = model.__class__.__name__

    with open(filename, "w") as f:
        json.dump(model_data, f, cls=MlxtendEncoder, indent=4)


def load_model_from_json(filename):
    """Load an mlxtend estimator from a JSON file."""
    with open(filename, "r") as f:
        model_data = json.load(f)
    module_name = model_data.pop("__module__")
    class_name = model_data.pop("__class__")

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    model_instance = class_()

    for key, value in model_data.items():
        if isinstance(value, list) and key.endswith("_"):
            setattr(model_instance, key, np.array(value))
        else:
            setattr(model_instance, key, value)
    return model_instance
