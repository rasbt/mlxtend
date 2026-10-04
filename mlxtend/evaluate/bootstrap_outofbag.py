# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Bootstrap functions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class BootstrapOutOfBag(object):
    """
    Parameters
    ----------

    n_splits : int (default=200)
        Number of bootstrap iterations.
        Must be larger than 1.

    random_seed : int (default=None)
        If int, random_seed is the seed used by
        the random number generator.


    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/BootstrapOutOfBag/

    """

    def __init__(self, n_splits=200, random_seed=None):
        self.random_seed = random_seed

        if not isinstance(n_splits, int) or n_splits < 1:
            raise ValueError("Number of splits must be greater than 1.")
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        """Generate bootstrap training and out-of-bag test indices.

        Parameters
        ----------
        X : array-like, shape (n_samples, ...)
            Data to split. Only the number of samples in ``X.shape[0]`` is used.

        y : array-like or None (default: None)
            Argument is not used and only included as parameter
            for compatibility, similar to `KFold` in scikit-learn.

        groups : array-like or None (default: None)
            Argument is not used and only included as parameter
            for compatibility, similar to `KFold` in scikit-learn.

        Yields
        ------
        train_idx : ndarray
            Training indices sampled with replacement. Indices may repeat.

        test_idx : ndarray
            Indices not selected in the training sample.

        """
        rng = np.random.RandomState(self.random_seed)
        sample_idx = np.arange(X.shape[0])
        set_idx = set(sample_idx)

        for _ in range(self.n_splits):
            train_idx = rng.choice(sample_idx, size=sample_idx.shape[0], replace=True)
            test_idx = np.array(list(set_idx - set(train_idx)))
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility with scikit-learn.

        y : object
            Always ignored, exists for compatibility with scikit-learn.

        groups : object
            Always ignored, exists for compatibility with scikit-learn.

        Returns
        -------

        n_splits : int
            Returns the number of splitting iterations in the cross-validator.

        """
        return self.n_splits
