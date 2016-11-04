# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A function for plotting learning curves of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings


def plot_learning_curves(*args, **kwargs):
    """

    Note that importing this function from mlxtend.evaluate has been
    deprecated and will not longer be supported in mlxtend 0.6.
    Please use `from mlxtend.plotting import plot_learning_curves` instead.

    """

    from ..plotting import plot_learning_curves

    warnings.warn("Note that importing this function from mlxtend.evaluate"
                  " has been deprecated and will not longer be supported in"
                  " mlxtend 0.6. Please use"
                  "`from mlxtend.plotting import plot_learning_curves` "
                  "instead.",
                  DeprecationWarning,
                  stacklevel=2)

    return plot_learning_curves(args, kwargs)
