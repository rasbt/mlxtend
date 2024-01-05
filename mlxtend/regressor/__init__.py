# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .linear_regression import LinearRegression
from .stacking_cv_regression import StackingCVRegressor
from .stacking_regression import StackingRegressor

__all__ = ["LinearRegression", "StackingRegressor", "StackingCVRegressor"]
