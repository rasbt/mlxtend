# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Base Cluster (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from collections import defaultdict
from time import time

try:
    from inspect import signature
except ImportError:
    from ..externals.signature_py27 import signature


class _BaseModel(object):
    def __init__(self):
        self._init_time = time()

    def _check_arrays(self, X, y=None):
        if isinstance(X, list):
            raise ValueError("X must be a numpy array")
        if not len(X.shape) == 2:
            raise ValueError("X must be a 2D array. Try X[:, numpy.newaxis]")
        try:
            if y is None:
                return
        except AttributeError:
            if not len(y.shape) == 1:
                raise ValueError("y must be a 1D array.")

        if not len(y) == X.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the estimator

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 4 clause



        """
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "scikit-learn estimators should always "
                    "specify their parameters in the signature"
                    " of their __init__ (no varargs)."
                    " %s with constructor %s doesn't "
                    " follow this convention." % (cls, init_signature)
                )
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.'

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 3 clause

        """
        out = dict()
        for key in self._get_param_names():
            value = getattr(self, key, None)
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                out.update((key + "__" + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.
        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self

        adapted from
        https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/base.py
        Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
        License: BSD 3 clause

        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        nested_params = defaultdict(dict)  # grouped by prefix
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                raise ValueError(
                    "Invalid parameter %s for estimator %s. "
                    "Check the list of available parameters "
                    "with `estimator.get_params().keys()`." % (key, self)
                )

            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value

        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)

        return self
