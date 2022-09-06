import numpy as np
from sklearn.model_selection import cross_val_score


def _merge_lists(nested_list, high_level_indices=None):
    """
    merge elements of lists (of a nested_list) into one single tuple with elements
    sorted in ascending order.

    Parameters
    ----------
    nested_list: List
        a  list whose elements must be list as well.

    high_level_indices: list or tuple, default None
        a list or tuple that contains integers that are between 0 (inclusive) and
        the length of `nested_lst` (exclusive). If None, the merge of all
        lists nested in `nested_list` will be returned.

    Returns
    -------
    out: tuple
        a tuple, with elements sorted in ascending order, that is the merge of inner
        lists whose indices are provided in `high_level_indices`

    Example:
    nested_list = [[1],[2, 3],[4]]
    high_level_indices = [1, 2]
    >>> _merge_lists(nested_list, high_level_indices)
    (2, 3, 4) # merging [2, 3] and [4]
    """
    if high_level_indices is None:
        high_level_indices = list(range(len(nested_list)))

    out = []
    for idx in high_level_indices:
        out.extend(nested_list[idx])

    return tuple(sorted(out))


def _calc_score(
    selector, X, y, indices, groups=None, feature_groups=None, **fit_params
):
    """
    calculate the cross-validation score for feature data `X` and target variable
     `y`.

    Parameters
    ---------
    selector : objcet with attributes est_` (estimator), `cv` (number of folds
    in cross-validation), and `pre_dispatch`()

    X : numpy.ndarray
        A 2D array consisting of feature data, where each column corresponds to
        one feature, and each row corresponds to one instance (or observation)

    y : numpy.ndarray
        A 1D array consiting of tartget values

    indices :

    groups :

    feature_groups :

    **fit_params :

    Returns
    -------
    indices :

    scores :
    """
    if feature_groups is None:
        feature_groups = [[i] for i in range(X.shape[1])]

    IDX = _merge_lists(feature_groups, indices)
    if selector.cv:
        scores = cross_val_score(
            selector.est_,
            X[:, IDX],
            y,
            groups=groups,
            cv=selector.cv,
            scoring=selector.scorer,
            n_jobs=1,
            pre_dispatch=selector.pre_dispatch,
            fit_params=fit_params,
        )
    else:
        selector.est_.fit(X[:, IDX], y, **fit_params)
        scores = np.array([selector.scorer(selector.est_, X[:, IDX], y)])
    return indices, scores


def _preprocess(X):
    """
    Check if X is a DataFrame or not, and returns numpy ndarray and name of features.

    Parameters
    ----------
    X : DataFrame or numpy.ndarray
        A DataFrame or a 2D numpy.ndarray

    Returns
    -------
    X_ : numpy.ndarray
        A 2D array that is equivalanet to X.to_numpy()

    features_names : List
        A list consisting of name of features. When `X` is a DataFrame, it contains
        the name of columns. If it is a 2D array, features_names[i] is str(i).
    """
    if X.ndim != 2:
        raise ValueError(f"The input X must be 2D array. Got {X.ndim}")

    if type(X).__name__ == "DataFrame":
        features_names = list(X.columns)
        X_ = X.to_numpy(copy=True)
    else:
        # it is numpy array
        features_names = [str(i) for i in range(X.shape[1])]
        X_ = X.copy()

    return X_, features_names
