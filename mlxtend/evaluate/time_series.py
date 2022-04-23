# mlxtend Machine Learning Library Extensions
#
# Time series cross validation with grouping.
# Author: Dmitry Labazkin <labdmitriy@gmail.com>
#
# License: BSD 3 clause

from itertools import groupby

import numpy as np
from sklearn.utils import indexable


class GroupTimeSeriesSplit:
    """Group time series cross-validator.

    Parameters
    ----------
    test_size : int
        Size of test dataset.
    train_size : int (default=None)
        Size of train dataset.
    n_splits : int (default=None)
        Number of the splits.
    gap_size : int (default=0)
        Gap size between train and test datasets.
    shift_size : int (default=1)
        Step to shift for the next fold.
    window_type : str (default='rolling')
        Type of the window. Possible values: 'rolling', 'expanding'.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/GroupTimeSeriesSplit/
    """

    def __init__(self,
                 test_size,
                 train_size=None,
                 n_splits=None,
                 gap_size=0,
                 shift_size=1,
                 window_type='rolling'):

        if (train_size is None) and (n_splits is None):
            raise ValueError(
                'Either train_size or n_splits have to be defined')

        if window_type not in ['rolling', 'expanding']:
            raise ValueError(
                'Window type can be either "rolling" or "expanding"')

        if (train_size is not None) and (window_type == 'expanding'):
            raise ValueError(
                'Train size can be specified only with rolling window')

        self.test_size = test_size
        self.train_size = train_size
        self.n_splits = n_splits
        self.gap_size = gap_size
        self.shift_size = shift_size
        self.window_type = window_type

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like (default=None)
            Always ignored, exists for compatibility.
        groups : array-like (default=None)
            Array with group names or sequence numbers.

        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        test_size = self.test_size
        gap = self.gap_size
        shift_size = self.shift_size
        X, y, groups = indexable(X, y, groups)

        if groups is None:
            raise ValueError('The groups should be specified')

        group_seqs = [group[0] for group in groupby(groups)]
        unique_groups, group_starts_idx = np.unique(groups, return_index=True)

        if group_seqs != sorted(unique_groups):
            raise ValueError('The groups should be sorted in increasing order')

        n_groups = len(unique_groups)
        self._n_groups = n_groups
        groups_dict = dict(zip(unique_groups, group_starts_idx))
        n_samples = len(X)

        self._calculate_split_params()
        train_size = self.train_size
        n_splits = self.n_splits
        train_start_idx = self._train_start_idx
        train_end_idx = train_start_idx + train_size
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_size

        for _ in range(n_splits):
            train_idx = np.r_[slice(groups_dict[group_seqs[train_start_idx]],
                                    groups_dict[group_seqs[train_end_idx]])]

            if test_end_idx < n_groups:
                test_idx = np.r_[slice(groups_dict[group_seqs[test_start_idx]],
                                       groups_dict[group_seqs[test_end_idx]])]
            else:
                test_idx = np.r_[slice(groups_dict[group_seqs[test_start_idx]],
                                       n_samples)]

            yield train_idx, test_idx

            if self.window_type == 'rolling':
                train_start_idx = train_start_idx + shift_size

            train_end_idx = train_end_idx + shift_size
            test_start_idx = test_start_idx + shift_size
            test_end_idx = test_end_idx + shift_size

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator.

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.
        y : object
            Always ignored, exists for compatibility.
        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

    def _calculate_split_params(self):
        train_size = self.train_size
        test_size = self.test_size
        n_splits = self.n_splits
        gap = self.gap_size
        shift_size = self.shift_size
        n_groups = self._n_groups

        not_enough_data_error = (
            'Not enough data to split number of groups ({0})'
            ' for number splits ({1})'
            ' with train size ({2}), test size ({3}),'
            ' gap size ({4}), shift size ({5})')

        if (train_size is None) and (n_splits is not None):
            train_size = n_groups - gap - test_size - (n_splits -
                                                       1) * shift_size
            self.train_size = train_size

            if train_size <= 0:
                raise ValueError(
                    not_enough_data_error.format(n_groups, n_splits,
                                                 train_size, test_size, gap,
                                                 shift_size))
            train_start_idx = 0
        elif (n_splits is None) and (train_size is not None):
            n_splits = (n_groups - train_size - gap -
                        test_size) // shift_size + 1
            self.n_splits = n_splits

            if self.n_splits <= 0:
                raise ValueError(
                    not_enough_data_error.format(n_groups, n_splits,
                                                 train_size, test_size, gap,
                                                 shift_size))
            train_start_idx = n_groups - train_size - gap - test_size - (
                n_splits - 1) * shift_size
        else:
            train_start_idx = n_groups - train_size - gap - test_size - (
                n_splits - 1) * shift_size

            if train_start_idx < 0:
                raise ValueError(
                    not_enough_data_error.format(n_groups, n_splits,
                                                 train_size, test_size, gap,
                                                 shift_size))

        self._train_start_idx = train_start_idx
