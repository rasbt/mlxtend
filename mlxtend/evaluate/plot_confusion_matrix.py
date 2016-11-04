# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# A function for plotting a confusion matrix.
# License: BSD 3 clause

import warnings


def plot_confusion_matrix(*args, **kwargs):
    """

    Note that importing this function from mlxtend.evaluate has been
    deprecated and will not longer be supported in mlxtend 0.6.
    Please use `from mlxtend.plotting import plot_confusion_matrix` instead.

    """

    from ..plotting import plot_confusion_matrix

    warnings.warn("Note that importing this function from mlxtend.evaluate"
                  " has been deprecated and will not longer be supported in"
                  " mlxtend 0.6. Please use"
                  "`from mlxtend.plotting import plot_confusion_matrix` "
                  "instead.",
                  DeprecationWarning,
                  stacklevel=2)

    return plot_confusion_matrix(*args, **kwargs)
