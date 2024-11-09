# mlxtend Machine Learning Library Extensions
#
# Time series cross validation with grouping.
# Author: Dmitry Labazkin <labdmitriy@gmail.com>
#
# License: BSD 3 clause

from itertools import accumulate, chain, groupby, islice

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
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
    window_type : str (default="rolling")
        Type of the window. Possible values: "rolling", "expanding".

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/GroupTimeSeriesSplit/
    """

    def __init__(
        self,
        test_size,
        train_size=None,
        n_splits=None,
        gap_size=0,
        shift_size=1,
        window_type="rolling",
    ):
        if (train_size is None) and (n_splits is None):
            raise ValueError("Either train_size or n_splits should be defined")

        if window_type not in ["rolling", "expanding"]:
            raise ValueError('Window type can be either "rolling" or "expanding"')

        if (train_size is not None) and (window_type == "expanding"):
            raise ValueError("Train size can be specified only with rolling window")

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
            raise ValueError("The groups should be specified")

        group_names, group_lengths = zip(
            *[
                (group_name, len(list(group_seq)))
                for group_name, group_seq in groupby(groups)
            ]
        )
        n_groups = len(group_names)

        if n_groups != len(set(group_names)):
            raise ValueError("The groups should be consecutive")

        self._n_groups = n_groups
        group_starts_idx = chain(
            [0],
            islice(accumulate(group_lengths), len(group_lengths) - 1),
        )
        groups_dict = dict(zip(group_names, group_starts_idx))
        n_samples = len(X)

        self._calculate_split_params()
        train_size = self.train_size
        n_splits = self.n_splits
        train_start_idx = self._train_start_idx
        train_end_idx = train_start_idx + train_size
        test_start_idx = train_end_idx + gap
        test_end_idx = test_start_idx + test_size

        for _ in range(n_splits):
            train_idx = np.r_[
                slice(
                    groups_dict[group_names[train_start_idx]],
                    groups_dict[group_names[train_end_idx]],
                )
            ]

            if test_end_idx < n_groups:
                test_idx = np.r_[
                    slice(
                        groups_dict[group_names[test_start_idx]],
                        groups_dict[group_names[test_end_idx]],
                    )
                ]
            else:
                test_idx = np.r_[
                    slice(groups_dict[group_names[test_start_idx]], n_samples)
                ]

            yield train_idx, test_idx

            if self.window_type == "rolling":
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
            "Not enough data to split number of groups ({0})"
            " for number splits ({1})"
            " with train size ({2}), test size ({3}),"
            " gap size ({4}), shift size ({5})"
        )

        if (train_size is None) and (n_splits is not None):
            train_size = n_groups - gap - test_size - (n_splits - 1) * shift_size
            self.train_size = train_size

            if train_size <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups,
                        n_splits,
                        train_size,
                        test_size,
                        gap,
                        shift_size,
                    )
                )
            train_start_idx = 0
        elif (n_splits is None) and (train_size is not None):
            n_splits = (n_groups - train_size - gap - test_size) // shift_size + 1
            self.n_splits = n_splits

            if self.n_splits <= 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups,
                        n_splits,
                        train_size,
                        test_size,
                        gap,
                        shift_size,
                    )
                )
            train_start_idx = (
                n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size
            )
        else:
            train_start_idx = (
                n_groups - train_size - gap - test_size - (n_splits - 1) * shift_size
            )

            if train_start_idx < 0:
                raise ValueError(
                    not_enough_data_error.format(
                        n_groups,
                        n_splits,
                        train_size,
                        test_size,
                        gap,
                        shift_size,
                    )
                )

        self._train_start_idx = train_start_idx


def print_split_info(X, y, groups, **cv_args):
    """Print information details about splits."""
    cv = GroupTimeSeriesSplit(**cv_args)
    groups = np.array(groups)

    for train_idx, test_idx in cv.split(X, groups=groups):
        print("Train indices:", train_idx)
        print("Test indices:", test_idx)
        print("Train length:", len(train_idx))
        print("Test length:", len(test_idx))
        print("Train groups:", groups[train_idx])
        print("Test groups:", groups[test_idx])
        print("Train group size:", len(set(groups[train_idx])))
        print("Test group size:", len(set(groups[test_idx])))
        print("Train group months:", X.index[train_idx].values)
        print("Test group months:", X.index[test_idx].values)
        print()


def plot_split_indices(cv, cv_args, X, y, groups, n_splits, image_file_path=None):
    """Create a sample plot for indices of a cross-validation object."""
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap_data = plt.cm.tab20
    cmap_cv = plt.cm.coolwarm
    lw = 10
    marker_size = 200

    for split_idx, (train_idx, test_idx) in enumerate(
        cv.split(X=X, y=y, groups=groups)
    ):
        indices = np.array([np.nan] * len(X))
        indices[test_idx] = 1
        indices[train_idx] = 0

        ax.scatter(
            range(len(X)),
            [split_idx + 0.5] * len(X),
            c=indices,
            marker="_",
            lw=lw,
            cmap=cmap_cv,
            vmin=-0.4,
            vmax=1.4,
            s=marker_size,
        )

    ax.scatter(
        range(len(X)),
        [split_idx + 1.5] * len(X),
        c=groups,
        marker="_",
        lw=lw,
        cmap=cmap_data,
        s=marker_size,
    )

    yticklabels = list(range(1, n_splits + 1)) + ["group"]
    ax.set(
        yticks=np.arange(n_splits + 1) + 0.5,
        yticklabels=yticklabels,
        ylabel="CV iteration",
        ylim=[n_splits + 1.2, -0.2],
        xlim=[-0.5, len(indices) - 0.5],
    )

    legend_splits = ax.legend(
        [Patch(color=cmap_cv(0.2)), Patch(color=cmap_cv(0.8))],
        ["Training set", "Testing set"],
        title="Data Splits",
        loc="upper right",
        fontsize=13,
    )

    ax.add_artist(legend_splits)

    group_labels = [f"{group}" for group in np.unique(groups)]
    cmap = plt.cm.get_cmap("tab20", len(group_labels))

    unique_patches = {}
    for i, group in enumerate(np.unique(groups)):
        unique_patches[group] = Patch(color=cmap(i), label=f"{group}")

    ax.legend(
        handles=list(unique_patches.values()),
        title="Groups",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=13,
    )

    ax.set_title("{}\n{}".format(type(cv).__name__, cv_args), fontsize=15)
    ax.set_xlim(0, len(X))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel(xlabel="Sample index", fontsize=13)
    ax.set_ylabel(ylabel="CV iteration", fontsize=13)
    ax.tick_params(axis="both", which="major", labelsize=13)
    ax.tick_params(axis="both", which="minor", labelsize=13)

    plt.tight_layout()

    if image_file_path:
        plt.savefig(image_file_path, bbox_inches="tight")

    plt.show()


def plot_splits(X, y, groups, image_file_path=None, **cv_args):
    """Visualize splits by group."""
    cv = GroupTimeSeriesSplit(**cv_args)
    cv._n_groups = len(np.unique(groups))
    cv._calculate_split_params()
    n_splits = cv.n_splits

    plot_split_indices(
        cv, cv_args, X, y, groups, n_splits, image_file_path=image_file_path
    )


def print_cv_info(cv, X, y, groups, clf, scores):
    """Print information details about cross-validation usage with classifier."""
    for split_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_train_pred = clf.predict(X.iloc[train_idx])
        y_test_pred = clf.predict(X.iloc[test_idx])
        print(f"Split number: {split_idx + 1}")
        print(f"Train true target: {y.iloc[train_idx].values}")
        print(f"Train predicted target: {y_train_pred}")
        print(f"Test true target: {y.iloc[test_idx].values}")
        print(f"Test predicted target: {y_test_pred}")
        print(f"Accuracy: {scores[split_idx].round(2)}")
        print()
