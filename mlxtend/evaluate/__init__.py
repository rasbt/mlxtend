# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .scoring import scoring
from .confusion_matrix import confusion_matrix
from .mcnemar import mcnemar_table
from .mcnemar import mcnemar
from .decision_regions import plot_decision_regions
from .learning_curves import plot_learning_curves
from .plot_confusion_matrix import plot_confusion_matrix


__all__ = ["scoring", "confusion_matrix",
           "mcnemar_table", "mcnemar",
           "plot_decision_regions",  # deprecated
           "plot_confusion_matrix",  # deprecated
           "plot_learning_curves"]  # deprecated
