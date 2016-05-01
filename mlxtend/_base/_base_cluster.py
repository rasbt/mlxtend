# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Base Clusteer (Clutering Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from ._base_unsupervised_estimator import _BaseUnsupervisedEstimator


class _BaseCluster(_BaseUnsupervisedEstimator):

    """Parent Class Unsupervised Estimator

    A base class that is implemented by clustering estimators

    """
    def __init__(self, print_progress=0, random_seed=0):
        super(_BaseCluster, self).__init__(
            print_progress=print_progress,
            random_seed=random_seed)
