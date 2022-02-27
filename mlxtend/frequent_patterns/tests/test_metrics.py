import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.frequent_patterns import metrics
from numpy.testing import assert_raises as numpy_assert_raises


dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit_transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
df_freq_items_with_colnames = apriori(df, min_support=0.6, use_colnames=True)
df_strong_rules = association_rules(
    df_freq_items_with_colnames, metric="confidence", min_threshold=0.7)


def test_kulczynski_measure_default():
    a = frozenset(['Onion'])
    b = frozenset(['Kidney Beans', 'Eggs'])
    assert metrics.kulczynski_measure(df_strong_rules, a, b) == 0.875


def test_kulczynski_measure_set():
    a = set(['Onion'])
    b = set(['Kidney Beans', 'Eggs'])
    assert metrics.kulczynski_measure(df_strong_rules, a, b) == 0.875


def test_kulczynski_measure_no_antecedent():
    a = frozenset(['Laptop'])
    b = frozenset(['Kidney Beans', 'Eggs'])
    assert metrics.kulczynski_measure(df_strong_rules, a, b) == 0.0


def test_kulczynski_measure_no_consequent():
    a = frozenset(['Onion'])
    b = frozenset(['Laptop'])
    assert metrics.kulczynski_measure(df_strong_rules, a, b) == 0.0


def test_kulczynski_measure_no_rule():
    a = frozenset(['Onion'])
    b = frozenset(['Kidney Beans', 'Eggs'])
    numpy_assert_raises(
        ValueError, metrics.kulczynski_measure, pd.DataFrame(), a, b)


def test_imbalance_ratio_default():
    a = frozenset(['Onion'])
    b = frozenset(['Kidney Beans', 'Eggs'])
    assert metrics.imbalance_ratio(
        df_freq_items_with_colnames, a, b) == 0.2500000000000001


def test_imbalance_ratio_set():
    a = set(['Onion'])
    b = set(['Kidney Beans', 'Eggs'])
    assert metrics.imbalance_ratio(
        df_freq_items_with_colnames, a, b) == 0.2500000000000001


def test_imbalance_ratio_no_itemset_a():
    a = frozenset([])
    b = frozenset(['Laptop'])
    assert metrics.imbalance_ratio(df_freq_items_with_colnames, a, b) == 0.0


def test_imbalance_ratio_no_itemset_b():
    a = frozenset(['Laptop'])
    b = frozenset([])
    assert metrics.imbalance_ratio(df_freq_items_with_colnames, a, b) == 0.0


def test_imbalance_ratio_no_itemset_a_b():
    a = frozenset([])
    b = frozenset([])
    assert metrics.imbalance_ratio(df_freq_items_with_colnames, a, b) == 0.0


def test_imbalance_ratio_no_rule():
    a = frozenset(['Onion'])
    b = frozenset(['Kidney Beans', 'Eggs'])
    numpy_assert_raises(
        ValueError, metrics.imbalance_ratio, pd.DataFrame(), a, b)
