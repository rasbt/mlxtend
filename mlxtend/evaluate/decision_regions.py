# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A function for plotting decision regions of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings


def plot_decision_regions(*args, **kwargs):
    """Plot decision regions of a classifier.

    Note that importing this function from mlxtend.evaluate has been
    deprecated and will not longer be supported in mlxtend 0.6.
    Please use `from mlxtend.plotting import plot_decision_regions` instead.

    """

    from ..plotting import plot_decision_regions

    warnings.warn("Note that importing this function from mlxtend.evaluate"
                  " has been deprecated and will not longer be supported in"
                  " mlxtend 0.6. Please use"
                  "`from mlxtend.plotting import plot_decision_regions` "
                  "instead.",
                  DeprecationWarning,
                  stacklevel=2)

    return plot_decision_regions(*args, **kwargs)
