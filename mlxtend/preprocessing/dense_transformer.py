# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# A class for transforming sparse numpy arrays into dense arrays.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


class DenseTransformer(object):

    """Convert a sparse matrix into a dense matrix."""

    def __init__(self, some_param=True):
        pass

    def transform(self, X, y=None):
        return X.toarray()

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return X.toarray()

    def get_params(self, deep=True):
        return {'some_param': True}
