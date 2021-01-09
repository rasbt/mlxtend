# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .linear_regression import LinearRegression
from .stacking_regression import StackingRegressor
from .stacking_cv_regression import StackingCVRegressor

__all__ = ["LinearRegression", "StackingRegressor", "StackingCVRegressor"]
