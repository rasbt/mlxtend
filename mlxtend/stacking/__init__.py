# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .stacking import StackingRegressor, StackingClassifier
from .stacking_cv import StackingCVRegressor, StackingCVClassifier

__all__ = ["StackingRegressor", "StackingClassifier",
           "StackingCVRegressor", "StackingCVClassifier"]