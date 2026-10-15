import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal
from scipy import sparse
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlxtend.feature_selection import SequentialFeatureSelector as SFS


@pytest.mark.parametrize("k_features", [1, 3])
@pytest.mark.parametrize("cv", [0, 3])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize(
    "forward,floating", [(True, False), (False, False), (True, True), (False, True)]
)
def test_dataframe_column_transformer(cv, n_jobs, forward, floating, k_features):
    """Dtype selectors must receive DataFrames during both fit and scoring."""
    X = pd.DataFrame(
        {
            "amount": np.arange(24, dtype=float),
            "kind": pd.Categorical(["low", "high"] * 12),
            "constant": np.ones(24),
            "other": np.zeros(24),
        },
        index=np.arange(100, 124),
    )
    original = X.copy(deep=True)
    y = np.array([0, 1] * 12)
    model = make_pipeline(
        ColumnTransformer(
            [
                (
                    "numeric",
                    StandardScaler(),
                    make_column_selector(dtype_include=np.number),
                ),
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore"),
                    make_column_selector(dtype_include="category"),
                ),
            ]
        ),
        LogisticRegression(),
    )
    selector = SFS(
        model,
        k_features=k_features,
        cv=cv,
        n_jobs=n_jobs,
        forward=forward,
        floating=floating,
        scoring="accuracy",
        preserve_dataframe=True,
    )
    selected = selector.fit_transform(X, y)
    assert "kind" in selector.k_feature_names_
    assert len(selector.k_feature_idx_) == k_features
    assert selector.k_score_ == 1.0
    assert_frame_equal(selected, X.iloc[:, list(selector.k_feature_idx_)])
    assert_frame_equal(X, original)
    assert clone(model).fit(selected, y).score(selected, y) == 1.0


class IndexClassifier(ClassifierMixin, BaseEstimator):
    """Check column dtypes and sample alignment inside actual CV workers."""

    def fit(self, X, y, sample_weight=None):
        assert isinstance(X, pd.DataFrame)
        assert X.dtypes.eq(np.dtype("int64")).all()
        assert_array_equal(X.index.to_numpy() % 2, y)
        assert_array_equal(sample_weight, X.index.to_numpy() + 1)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.index.to_numpy() % 2


@pytest.mark.parametrize("cv", [0, GroupKFold(3)])
@pytest.mark.parametrize("n_jobs", [1, 2])
@pytest.mark.parametrize("forward", [True, False])
def test_dataframe_groups_fixed_features_and_fit_params(cv, n_jobs, forward):
    X = pd.DataFrame(
        np.arange(96).reshape(24, 4),
        index=np.arange(100, 124),
        columns=["a", "b", "c", "d"],
    )
    selector = SFS(
        IndexClassifier(),
        k_features=2,
        cv=cv,
        n_jobs=n_jobs,
        forward=forward,
        scoring="accuracy",
        preserve_dataframe=True,
        feature_groups=[["a", "c"], ["b"], ["d"]],
        fixed_features=["a", "c"],
    )
    selector.fit(
        X,
        X.index.to_numpy() % 2,
        groups=np.repeat(np.arange(6), 4),
        sample_weight=X.index.to_numpy() + 1,
    )
    assert selector.k_score_ == 1.0
    assert {"a", "c"}.issubset(selector.k_feature_names_)
    assert_frame_equal(selector.transform(X), X.iloc[:, list(selector.k_feature_idx_)])


@pytest.mark.parametrize("preserve_dataframe", [False, True])
@pytest.mark.parametrize("kind", ["array", "dataframe", "csr", "csc"])
def test_dataframe_option_numeric_equivalence(preserve_dataframe, kind):
    y = np.array([0, 1] * 12)
    values = np.column_stack([y, np.ones(24)])
    X = {
        "array": lambda: values,
        "dataframe": lambda: pd.DataFrame(values, columns=["signal", "constant"]),
        "csr": lambda: sparse.csr_matrix(values),
        "csc": lambda: sparse.csc_matrix(values),
    }[kind]()
    selector = SFS(
        LogisticRegression(),
        k_features=1,
        cv=3,
        scoring="accuracy",
        preserve_dataframe=preserve_dataframe,
    )
    selected = selector.fit_transform(X, y)
    assert selector.k_feature_idx_ == (0,)
    assert selector.k_score_ == 1.0
    if kind == "dataframe" and preserve_dataframe:
        assert_frame_equal(selected, X[["signal"]])
    elif sparse.issparse(X):
        assert sparse.issparse(selected)
        assert_array_equal(selected.toarray(), values[:, [0]])
    else:
        assert isinstance(selected, np.ndarray)
        assert_array_equal(selected, values[:, [0]])


@pytest.mark.parametrize(
    "columns",
    [[20, 10], ["same", "same"], pd.MultiIndex.from_tuples([("x", 1), ("y", 2)])],
)
def test_dataframe_column_selection_is_positional(columns):
    class FrameSubclass(pd.DataFrame):
        @property
        def _constructor(self):
            return FrameSubclass

    X = FrameSubclass(
        np.arange(48).reshape(24, 2), index=np.arange(100, 124), columns=columns
    )
    selector = SFS(
        IndexClassifier(),
        k_features=1,
        cv=0,
        scoring="accuracy",
        preserve_dataframe=True,
        fixed_features=(1,),
    )
    selector.fit(X, X.index.to_numpy() % 2, sample_weight=X.index.to_numpy() + 1)
    assert_frame_equal(selector.transform(X), X.iloc[:, [1]])


def test_dataframe_option_clone_and_set_params():
    selector = SFS(LogisticRegression(), preserve_dataframe=True)
    assert clone(selector).get_params()["preserve_dataframe"] is True
    selector.set_params(preserve_dataframe=False)
    assert selector.preserve_dataframe is False


def test_dataframe_default_keeps_numpy_estimator_compatibility():
    from mlxtend.classifier import SoftmaxRegression

    y = np.array([0, 1] * 12)
    X = pd.DataFrame({"signal": y, "constant": np.ones(24)})
    options = dict(k_features=1, scoring="accuracy", cv=0)
    default = SFS(SoftmaxRegression(random_seed=1), **options).fit(X, y)
    array = SFS(SoftmaxRegression(random_seed=1), **options).fit(X.to_numpy(), y)
    assert default.k_feature_idx_ == array.k_feature_idx_
    assert_allclose(default.k_score_, array.k_score_)
    assert isinstance(default.transform(X), np.ndarray)
