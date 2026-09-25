import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .apriori import apriori
from .association_rules import association_rules


class RuleExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, min_support=0.1, metric="confidence", min_threshold=0.8):
        self.min_support = min_support
        self.metric = metric
        self.min_threshold = min_threshold

    def fit(self, X, y=None):
        X_df = (
            X.astype(bool)
            if isinstance(X, pd.DataFrame)
            else pd.DataFrame(X).astype(bool)
        )
        self.frequent_itemsets_ = apriori(
            X_df, min_support=self.min_support, use_colnames=True
        )
        return self

    def transform(self, X):
        if self.frequent_itemsets_.empty:
            return pd.DataFrame(
                columns=["antecedents", "consequents"]
                + ["support", "confidence", "lift", "leverage", "conviction"]
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            return association_rules(
                self.frequent_itemsets_,
                metric=self.metric,
                min_threshold=self.min_threshold,
            )
