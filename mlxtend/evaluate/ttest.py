# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import _safe_indexing, check_consistent_length
from sklearn.utils._tags import get_tags


def _infer_estimator_type(estimator):
    try:
        return get_tags(estimator).estimator_type
    except Exception:
        return getattr(estimator, "_estimator_type", None)


def paired_ttest_resampled(
    estimator1,
    estimator2,
    X,
    y,
    num_rounds=30,
    test_size=0.3,
    scoring=None,
    random_seed=None,
):
    """
    Implements the resampled paired t test procedure
    to compare the performance of two models
    (also called k-hold-out paired t test).

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    num_rounds : int (default: 30)
        Number of resampling iterations
        (i.e., train/test splits)

    test_size : float or int (default: 0.3)
        If float, should be between 0.0 and 1.0 and
        represent the proportion of the dataset to use
        as a test set.
        If int, represents the absolute number of test exsamples.

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.

    Returns
    ----------
    t : float
        The t-statistic

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/

    """
    if not isinstance(test_size, int) and not isinstance(test_size, float):
        raise ValueError(
            "train_size must be of " "type int or float. Got %s." % type(test_size)
        )

    rng = np.random.RandomState(random_seed)

    if scoring is None:
        est_type = _infer_estimator_type(estimator1)
        if est_type == "classifier":
            scoring = "accuracy"
        elif est_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must " "be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    score_diff = []
    for i in range(num_rounds):
        randint = rng.randint(low=0, high=32767)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=randint
        )

        estimator1.fit(X_train, y_train)
        estimator2.fit(X_train, y_train)

        est1_score = scorer(estimator1, X_test, y_test)
        est2_score = scorer(estimator2, X_test, y_test)
        score_diff.append(est1_score - est2_score)

    avg_diff = np.mean(score_diff)

    numerator = avg_diff * np.sqrt(num_rounds)
    denominator = np.sqrt(
        sum([(diff - avg_diff) ** 2 for diff in score_diff]) / (num_rounds - 1)
    )
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), num_rounds - 1) * 2.0
    return float(t_stat), float(pvalue)


def paired_ttest_corrected_resampled(
    estimator1,
    estimator2,
    X,
    y,
    num_rounds=30,
    test_size=0.3,
    cv=None,
    scoring=None,
    random_seed=None,
):
    """
    Implements the corrected resampled paired t test by
    Nadeau and Bengio (2003) to compare the performance of two models.

    The standard resampled paired t test treats the score differences
    from overlapping train/test splits as independent, which
    underestimates their variance and leads to too many false
    positives. The corrected test replaces the variance factor 1/J
    with (1/J + n_test/n_train), where J is the number of splits.
    Used with repeated k-fold splits, this is the corrected repeated
    k-fold cross-validation test by Bouckaert and Frank (2004).

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    num_rounds : int (default: 30)
        Number of resampling iterations
        (i.e., train/test splits). Ignored if `cv` is not None.

    test_size : float or int (default: 0.3)
        If float, should be between 0.0 and 1.0 and
        represent the proportion of the dataset to use
        as a test set.
        If int, represents the absolute number of test examples.
        Ignored if `cv` is not None.

    cv : scikit-learn cross-validation splitter or None (default: None)
        If None (default), uses `num_rounds` random train/test splits
        generated exactly as in `paired_ttest_resampled` for the same
        `random_seed`. Otherwise, an object with a `split(X, y)` method,
        for example `StratifiedShuffleSplit` or `RepeatedStratifiedKFold`,
        that yields at least two train/test splits.

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.
        Ignored if `cv` is not None.

    Returns
    ----------
    t : float
        The corrected t-statistic. If the score difference is identical
        in every split, the variance is zero: t is 0.0 if that difference
        is zero and +/- inf otherwise.

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Notes
    ----------
    Both estimators are cloned and refit on the training part of every
    split, so the estimators passed in are left unchanged and test data
    is never used for fitting. Put preprocessing steps such as scaling
    inside a scikit-learn `Pipeline` so that they are fit on the
    training part only.

    The references for both tests are listed in the user guide (see
    the link below).

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_corrected_resampled/

    """
    check_consistent_length(X, y)
    n_samples = len(y)

    if cv is None:
        if isinstance(num_rounds, bool) or not isinstance(
            num_rounds, (int, np.integer)
        ):
            raise ValueError(
                "num_rounds must be an integer. Got %s." % type(num_rounds)
            )
        if num_rounds < 2:
            raise ValueError("num_rounds must be at least 2. Got %d." % num_rounds)
        if isinstance(test_size, bool) or not isinstance(
            test_size, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                "test_size must be of type int or float. Got %s." % type(test_size)
            )

        # Same split generation as paired_ttest_resampled, applied to the
        # sample indices so that X and y can be indexed afterwards.
        rng = np.random.RandomState(random_seed)
        indices = np.arange(n_samples)
        splits = []
        for _ in range(num_rounds):
            randint = rng.randint(low=0, high=32767)
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=randint
            )
            splits.append((train_idx, test_idx))
    else:
        if not hasattr(cv, "split"):
            raise ValueError(
                "cv must be None or a cross-validation splitter with a "
                "split(X, y) method. Got %s." % type(cv)
            )
        splits = list(cv.split(X, y))
        if len(splits) < 2:
            raise ValueError(
                "cv must generate at least 2 train/test splits. Got %d." % len(splits)
            )

    if scoring is None:
        est_type = _infer_estimator_type(estimator1)
        if est_type == "classifier":
            scoring = "accuracy"
        elif est_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    score_diff = []
    test_train_ratio = []
    for train_idx, test_idx in splits:
        X_train, X_test = _safe_indexing(X, train_idx), _safe_indexing(X, test_idx)
        y_train, y_test = _safe_indexing(y, train_idx), _safe_indexing(y, test_idx)

        est1 = clone(estimator1).fit(X_train, y_train)
        est2 = clone(estimator2).fit(X_train, y_train)

        score_diff.append(scorer(est1, X_test, y_test) - scorer(est2, X_test, y_test))
        test_train_ratio.append(len(test_idx) / len(train_idx))

    score_diff = np.asarray(score_diff, dtype=float)
    num_splits = score_diff.shape[0]
    avg_diff = np.mean(score_diff)
    var_diff = np.var(score_diff, ddof=1)

    if var_diff == 0.0:
        if avg_diff == 0.0:
            return 0.0, 1.0
        return float(np.copysign(np.inf, avg_diff)), 0.0

    # Nadeau & Bengio's correction: 1/J becomes 1/J + n_test/n_train.
    # The ratio is averaged in case the split sizes differ slightly,
    # e.g., for k-fold splits when n_samples is not divisible by k.
    corrected_var = (1.0 / num_splits + np.mean(test_train_ratio)) * var_diff
    t_stat = avg_diff / np.sqrt(corrected_var)

    pvalue = stats.t.sf(np.abs(t_stat), num_splits - 1) * 2.0
    return float(t_stat), float(pvalue)


def paired_ttest_kfold_cv(
    estimator1, estimator2, X, y, cv=10, scoring=None, shuffle=False, random_seed=None
):
    """
    Implements the k-fold paired t test procedure
    to compare the performance of two models.

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    cv : int (default: 10)
        Number of splits and iteration for the
        cross-validation procedure

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    shuffle : bool (default: False)
        Whether to shuffle the dataset for generating
        the k-fold splits.

    random_seed : int or None (default: None)
        Random seed for shuffling the dataset
        for generating the k-fold splits.
        Ignored if shuffle=False.

    Returns
    ----------
    t : float
        The t-statistic

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_kfold_cv/

    """

    if not shuffle:
        kf = KFold(n_splits=cv, shuffle=shuffle)
    else:
        kf = KFold(n_splits=cv, random_state=random_seed, shuffle=shuffle)

    if scoring is None:
        est_type = _infer_estimator_type(estimator1)
        if est_type == "classifier":
            scoring = "accuracy"
        elif est_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    score_diff = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        estimator1.fit(X_train, y_train)
        estimator2.fit(X_train, y_train)

        est1_score = scorer(estimator1, X_test, y_test)
        est2_score = scorer(estimator2, X_test, y_test)
        score_diff.append(est1_score - est2_score)

    avg_diff = np.mean(score_diff)

    numerator = avg_diff * np.sqrt(cv)
    denominator = np.sqrt(
        sum([(diff - avg_diff) ** 2 for diff in score_diff]) / (cv - 1)
    )
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), cv - 1) * 2.0
    return float(t_stat), float(pvalue)


def paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None):
    """
    Implements the 5x2cv paired t test proposed
    by Dieterrich (1998)
    to compare the performance of two models.

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.

    Returns
    ----------
    t : float
        The t-statistic

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/

    """
    rng = np.random.RandomState(random_seed)

    if scoring is None:
        est_type = _infer_estimator_type(estimator1)
        if est_type == "classifier":
            scoring = "accuracy"
        elif est_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    variance_sum = 0.0
    first_diff = None

    def score_diff(X_1, X_2, y_1, y_2):
        estimator1.fit(X_1, y_1)
        estimator2.fit(X_1, y_1)
        est1_score = scorer(estimator1, X_2, y_2)
        est2_score = scorer(estimator2, X_2, y_2)
        score_diff = est1_score - est2_score
        return score_diff

    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=randint)

        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1

    numerator = first_diff
    denominator = np.sqrt(1 / 5.0 * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(pvalue)
