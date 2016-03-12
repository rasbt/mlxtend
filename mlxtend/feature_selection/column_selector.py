# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Object for selecting a dataset column in scikit-learn pipelines.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


class ColumnSelector(object):
    """Select specific columns from a data set.

    A feature selector for scikit-learn's Pipeline class that returns
    specified columns from a numpy array.

    """
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X, y=None):
        return X[:, self.cols]

    def fit(self, X, y=None):
        return self
