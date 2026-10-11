# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted

from mlxtend.data import boston_housing_data, iris_data
from mlxtend.evaluate import paired_ttest_corrected_resampled, paired_ttest_resampled
from mlxtend.feature_selection import ColumnSelector
from mlxtend.utils import assert_raises


class _SequenceScorer:
    """Scorer stub: returns the next value of `diffs` for estimator1
    and 0.0 for estimator2, so the score differences are known."""

    def __init__(self, diffs):
        self.diffs = list(diffs)
        self.calls = 0

    def __call__(self, estimator, X, y):
        split, which = divmod(self.calls, 2)
        self.calls += 1
        return self.diffs[split] if which == 0 else 0.0


def _models():
    return (
        LogisticRegression(random_state=1),
        DecisionTreeClassifier(random_state=1),
    )


def test_matches_formula_with_known_differences():
    X, y = iris_data()
    diffs = [0.05, 0.02, -0.01, 0.04, 0.03, 0.00, 0.06, 0.01, 0.02, 0.03]
    clf1, clf2 = _models()

    t, p = paired_ttest_corrected_resampled(
        clf1,
        clf2,
        X,
        y,
        num_rounds=len(diffs),
        test_size=0.3,
        scoring=_SequenceScorer(diffs),
        random_seed=1,
    )

    J = len(diffs)
    ratio = 45 / 105  # 30% of 150 samples for testing
    expected_t = np.mean(diffs) / np.sqrt((1 / J + ratio) * np.var(diffs, ddof=1))
    expected_p = 2 * stats.t.sf(abs(expected_t), J - 1)

    assert t == pytest.approx(expected_t)
    assert p == pytest.approx(expected_p)


def test_same_splits_as_uncorrected_resampled_test():
    X, y = iris_data()
    clf1, clf2 = _models()

    t_unc, p_unc = paired_ttest_resampled(clf1, clf2, X, y, random_seed=1)
    t_cor, p_cor = paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=1)

    # With identical splits, only the variance factor differs.
    J, ratio = 30, 45 / 105
    assert t_cor == pytest.approx(t_unc * np.sqrt((1 / J) / (1 / J + ratio)))
    assert abs(t_cor) < abs(t_unc)
    assert p_cor > p_unc


def test_classifier_defaults():
    X, y = iris_data()
    clf1, clf2 = _models()

    t, p = paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=1)
    assert round(t, 3) == 0.971, t
    assert round(p, 3) == 0.339, p

    # a clearly worse model is still detected after the correction
    clf2 = DecisionTreeClassifier(max_depth=1, random_state=1)
    t, p = paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=1)
    assert round(t, 3) == 11.407, t
    assert p < 0.001, p


def test_scoring_string_and_callable():
    X, y = iris_data()
    clf1, clf2 = _models()

    t, p = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, scoring="f1_macro", random_seed=1
    )
    assert round(t, 3) == 0.998, t
    assert round(p, 3) == 0.327, p

    scorer = make_scorer(recall_score, average="macro")
    t_callable, p_callable = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, scoring=scorer, random_seed=1
    )
    t_str, p_str = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, scoring="recall_macro", random_seed=1
    )
    assert t_callable == pytest.approx(t_str)
    assert p_callable == pytest.approx(p_str)


def test_regressor_defaults_to_r2():
    X, y = boston_housing_data()
    reg1 = Lasso(random_state=1)
    reg2 = Ridge(random_state=1)

    t, p = paired_ttest_corrected_resampled(reg1, reg2, X, y, random_seed=1)
    t_r2, p_r2 = paired_ttest_corrected_resampled(
        reg1, reg2, X, y, scoring="r2", random_seed=1
    )

    assert round(t, 3) == -2.066, t
    assert round(p, 3) == 0.048, p
    assert t == pytest.approx(t_r2)
    assert p == pytest.approx(p_r2)


def test_repeated_stratified_kfold_uses_k_minus_one_ratio():
    X, y = iris_data()
    clf1, clf2 = _models()
    diffs = [0.03, 0.01, 0.02, -0.01, 0.04, 0.00, 0.02, 0.05, 0.01, 0.03]
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=0)

    t, p = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, cv=cv, scoring=_SequenceScorer(diffs)
    )

    # Bouckaert & Frank's corrected repeated k-fold test: ratio = 1/(k-1)
    J, ratio = 10, 1 / (5 - 1)
    expected_t = np.mean(diffs) / np.sqrt((1 / J + ratio) * np.var(diffs, ddof=1))
    assert t == pytest.approx(expected_t)
    assert p == pytest.approx(2 * stats.t.sf(abs(expected_t), J - 1))


def test_stratified_shuffle_split_and_reproducibility():
    X, y = iris_data()
    clf1, clf2 = _models()

    def run():
        cv = StratifiedShuffleSplit(n_splits=20, test_size=0.25, random_state=0)
        return paired_ttest_corrected_resampled(clf1, clf2, X, y, cv=cv)

    t, p = run()
    assert np.isfinite(t)
    assert 0.0 <= p <= 1.0
    assert (t, p) == run()


def test_random_seed_reproducible_including_zero():
    X, y = iris_data()
    clf1, clf2 = _models()

    for seed in (0, 1):
        first = paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=seed)
        second = paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=seed)
        assert first == second

    assert paired_ttest_corrected_resampled(
        clf1, clf2, X, y, random_seed=0
    ) != paired_ttest_corrected_resampled(clf1, clf2, X, y, random_seed=1)


def test_returns_python_floats():
    X, y = iris_data()
    clf1, clf2 = _models()

    t, p = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, num_rounds=5, random_seed=1
    )
    assert type(t) is float
    assert type(p) is float


def test_input_estimators_are_not_fitted():
    X, y = iris_data()
    clf1, clf2 = _models()

    paired_ttest_corrected_resampled(clf1, clf2, X, y, num_rounds=5, random_seed=1)

    for clf in (clf1, clf2):
        with pytest.raises(NotFittedError):
            check_is_fitted(clf)


def test_pipelines_with_column_selector():
    X, y = iris_data()
    pipe1 = make_pipeline(
        ColumnSelector(cols=(0, 1)), StandardScaler(), LogisticRegression()
    )
    pipe2 = make_pipeline(
        ColumnSelector(cols=(2, 3)), StandardScaler(), LogisticRegression()
    )

    t, p = paired_ttest_corrected_resampled(
        pipe1, pipe2, X, y, num_rounds=10, random_seed=1
    )

    # petal features (2, 3) separate iris far better than sepal features
    assert t < 0
    assert p < 0.05


def test_pandas_input_matches_numpy_input():
    X, y = iris_data()
    clf1, clf2 = _models()

    result_np = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, num_rounds=5, random_seed=1
    )
    result_pd = paired_ttest_corrected_resampled(
        clf1, clf2, pd.DataFrame(X), pd.Series(y), num_rounds=5, random_seed=1
    )
    assert result_np == result_pd


def test_sparse_and_list_input_match_numpy_input():
    X, y = iris_data()
    clf1 = LogisticRegression(max_iter=1000)
    clf2 = LogisticRegression(C=0.01, max_iter=1000)

    result_np = paired_ttest_corrected_resampled(
        clf1, clf2, X, y, num_rounds=5, random_seed=1
    )
    result_sparse = paired_ttest_corrected_resampled(
        clf1, clf2, csr_matrix(X), y, num_rounds=5, random_seed=1
    )
    result_list = paired_ttest_corrected_resampled(
        clf1, clf2, X.tolist(), y.tolist(), num_rounds=5, random_seed=1
    )
    assert result_sparse == pytest.approx(result_np)
    assert result_list == pytest.approx(result_np)


def test_identical_models_return_t_zero_p_one():
    X, y = iris_data()
    clf = DecisionTreeClassifier(random_state=1)

    t, p = paired_ttest_corrected_resampled(clf, clf, X, y, num_rounds=5, random_seed=1)
    assert (t, p) == (0.0, 1.0)


def test_constant_nonzero_difference_returns_infinite_t():
    X, y = iris_data()
    clf1, clf2 = _models()

    for diff in (0.1, -0.1):
        t, p = paired_ttest_corrected_resampled(
            clf1,
            clf2,
            X,
            y,
            num_rounds=4,
            scoring=_SequenceScorer([diff] * 4),
            random_seed=1,
        )
        assert t == np.copysign(np.inf, diff)
        assert p == 0.0


def test_invalid_inputs():
    X, y = iris_data()
    clf1, clf2 = _models()

    assert_raises(
        ValueError,
        "num_rounds must be at least 2. Got 1.",
        paired_ttest_corrected_resampled,
        clf1,
        clf2,
        X,
        y,
        num_rounds=1,
    )
    for bad_rounds in (2.5, True):
        assert_raises(
            ValueError,
            "num_rounds must be an integer.",
            paired_ttest_corrected_resampled,
            clf1,
            clf2,
            X,
            y,
            num_rounds=bad_rounds,
        )
    assert_raises(
        ValueError,
        "test_size must be of type int or float. Got <class 'NoneType'>.",
        paired_ttest_corrected_resampled,
        clf1,
        clf2,
        X,
        y,
        test_size=None,
    )
    assert_raises(
        ValueError,
        "cv must be None or a cross-validation splitter",
        paired_ttest_corrected_resampled,
        clf1,
        clf2,
        X,
        y,
        cv=5,
    )
    assert_raises(
        ValueError,
        "cv must generate at least 2 train/test splits. Got 1.",
        paired_ttest_corrected_resampled,
        clf1,
        clf2,
        X,
        y,
        cv=ShuffleSplit(n_splits=1, random_state=0),
    )
    assert_raises(
        ValueError,
        "inconsistent numbers of samples",
        paired_ttest_corrected_resampled,
        clf1,
        clf2,
        X,
        y[:-1],
    )


def test_unknown_estimator_type_requires_scoring():
    X, y = iris_data()

    class NotAnEstimator:
        def fit(self, X, y):
            return self

        def get_params(self, deep=True):
            return {}

    assert_raises(
        AttributeError,
        "Estimator must be a Classifier or Regressor.",
        paired_ttest_corrected_resampled,
        NotAnEstimator(),
        NotAnEstimator(),
        X,
        y,
        num_rounds=2,
    )
