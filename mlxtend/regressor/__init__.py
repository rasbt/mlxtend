# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .linear_regression import LinearRegression
from .stacking_regression import StackingRegressor
from .oof_stacking_regression import OutOfFoldStackingRegressor

__all__ = ["LinearRegression", "StackingRegressor",
           "OutOfFoldStackingRegressor"]
