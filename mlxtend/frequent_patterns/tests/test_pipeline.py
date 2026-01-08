import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from mlxtend.frequent_patterns import RuleExtractor


def test_rule_extractor_basic():
    data = pd.DataFrame([[1, 0, 1], [1, 1, 1], [0, 1, 1]], columns=["A", "B", "C"])

    pipe = Pipeline([("extractor", RuleExtractor(min_support=0.1))])

    rules = pipe.fit_transform(data)

    assert isinstance(rules, pd.DataFrame)
    assert not rules.empty
    assert "antecedents" in rules.columns
    assert "consequents" in rules.columns


def test_rule_extractor_empty():
    data = pd.DataFrame([[1, 0], [0, 1]], columns=["A", "B"])
    extractor = RuleExtractor(min_support=0.9)
    rules = extractor.fit_transform(data)
    assert rules.empty
