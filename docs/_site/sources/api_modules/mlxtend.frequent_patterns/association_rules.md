## association_rules

*association_rules(df, metric='confidence', min_threshold=0.8, support_only=False)*

Generates a DataFrame of association rules including the
metrics 'score', 'confidence', and 'lift'

**Parameters**

- `df` : pandas DataFrame

    pandas DataFrame of frequent itemsets
    with columns ['support', 'itemsets']


- `metric` : string (default: 'confidence')

    Metric to evaluate if a rule is of interest.
**Automatically set to 'support' if `support_only=True`.**
    Otherwise, supported metrics are 'support', 'confidence', 'lift',

'leverage', and 'conviction'
    These metrics are computed as follows:

    - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]

    - confidence(A->C) = support(A+C) / support(A), range: [0, 1]

    - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]

    - leverage(A->C) = support(A->C) - support(A)*support(C),
    range: [-1, 1]

    - conviction = [1 - support(C)] / [1 - confidence(A->C)],
    range: [0, inf]



- `min_threshold` : float (default: 0.8)

    Minimal threshold for the evaluation metric,
    via the `metric` parameter,
    to decide whether a candidate rule is of interest.


- `support_only` : bool (default: False)

    Only computes the rule support and fills the other
    metric columns with NaNs. This is useful if:

    a) the input DataFrame is incomplete, e.g., does
    not contain support values for all rule antecedents
    and consequents

    b) you simply want to speed up the computation because
    you don't need the other metrics.

**Returns**

pandas DataFrame with columns "antecedents" and "consequents"
    that store itemsets, plus the scoring metric columns:
    "antecedent support", "consequent support",
    "support", "confidence", "lift",
    "leverage", "conviction"
    of all rules for which
    metric(rule) >= min_threshold.
    Each entry in the "antecedents" and "consequents" columns are
    of type `frozenset`, which is a Python built-in type that
    behaves similarly to sets except that it is immutable
    (For more info, see
    https://docs.python.org/3.6/library/stdtypes.html#frozenset).

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)

