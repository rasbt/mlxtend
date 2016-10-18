# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Algorithm for plotting sequential feature selection.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import warnings


def plot_sequential_feature_selection(**args):
    """

    Note that importing this function from mlxtend.evaluate has been
    deprecated and will not longer be supported in mlxtend 0.6.
    Please use `from mlxtend.plotting import plot_sequential_feature_selection`
    instead.

    """

    from ..plotting import plot_sequential_feature_selection

    warnings.warn("Note that importing this function from mlxtend.evaluate"
                  " has been deprecated and will not longer be supported in"
                  " mlxtend 0.6. Please use"
                  "`from mlxtend.plotting import"
                  "plot_sequential_feature_selection` instead.",
                  DeprecationWarning,
                  stacklevel=2)

    return plot_sequential_feature_selection(**args)
