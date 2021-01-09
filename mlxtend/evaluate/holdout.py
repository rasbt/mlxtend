# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


import numpy as np
from sklearn.model_selection import train_test_split


class RandomHoldoutSplit(object):
    """Train/Validation set splitter for sklearn's GridSearchCV etc.

    Provides train/validation set indices to split a dataset
    into train/validation sets using random indices.

    Parameters
    ----------
    valid_size : float (default: 0.5)
        Proportion of examples that being assigned as
        validation examples. 1-`valid_size` will then automatically
        be assigned as training set examples.
    random_seed : int (default: None)
        The random seed for splitting the data
        into training and validation set partitions.
    stratify : bool (default: False)
        True or False, whether to perform a stratified
        split or not

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/RandomHoldoutSplit/


    """

    def __init__(self, valid_size=0.5, random_seed=None, stratify=False):
        self.valid_size = valid_size
        self.random_seed = random_seed
        self.stratify = stratify

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (num_examples, num_features)
            Training data, where num_examples is the number of
            training examples and num_features is the number of features.

        y : array-like, shape (num_examples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train_index : ndarray
            The training set indices for that split.

        valid_index : ndarray
            The validation set indices for that split.
        """
        ind = np.arange(X.shape[0])
        if self.stratify:
            train_index, valid_index, _, _ = \
                    train_test_split(ind, y,
                                     test_size=self.valid_size,
                                     shuffle=True,
                                     stratify=y,
                                     random_state=self.random_seed)

        else:
            train_index, valid_index, _, _ = \
                    train_test_split(ind, y,
                                     test_size=self.valid_size,
                                     shuffle=True,
                                     stratify=y,
                                     random_state=self.random_seed)

        for i in range(1):
            yield train_index, valid_index

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : 1
            Returns the number of splitting iterations in the cross-validator.
            Always returns 1.
        """
        return 1


class PredefinedHoldoutSplit(object):
    """Train/Validation set splitter for sklearn's GridSearchCV etc.

    Uses user-specified train/validation set indices to split a dataset
    into train/validation sets using user-defined or random
    indices.

    Parameters
    ----------
    valid_indices : array-like, shape (num_examples,)
        Indices of the training examples in the training set
        to be used for validation. All other indices in the
        training set are used to for a training subset
        for model fitting.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/PredefinedHoldoutSplit/

    """

    def __init__(self, valid_indices):
        self.valid_indices = valid_indices

    def split(self, X, y, groups=None):
        """Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (num_examples, num_features)
            Training data, where num_examples is the number of examples
            and num_features is the number of features.

        y : array-like, shape (num_examples,)
            The target variable for supervised learning problems.
            Stratification is done based on the y labels.

        groups : object
            Always ignored, exists for compatibility.

        Yields
        ------
        train_index : ndarray
            The training set indices for that split.

        valid_index : ndarray
            The validation set indices for that split.
        """

        ind = np.arange(X.shape[0])
        train_mask = np.ones(X.shape[0], dtype=np.bool)
        train_mask[self.valid_indices] = False
        valid_mask = np.where(train_mask, False, True)

        for i in range(1):
            yield ind[train_mask], ind[valid_mask]

    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

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
        n_splits : 1
            Returns the number of splitting iterations in the cross-validator.
            Always returns 1.
        """
        return 1
