# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Function for generating association rules
#
# Author: Joshua Goerner <https://github.com/JoshuaGoerner>
#         Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from itertools import combinations
import numpy as np
import pandas as pd


def association_rules(df, metric="confidence", min_threshold=0.8):
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']
    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      Supported metrics are 'support', 'confidence', 'lift',
      'leverage', and 'conviction'
      These metrics are computed as follows:
      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]
      - leverage(A->C) = support(A->C) - support(A)*support(C), range: [-1, 1]
      - conviction = [1 - support(C)] / [1 - confidence(A->C)], range: [0, inf]
    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric
      to decide whether a candidate rule is of interest.

    Returns
    ----------
    pandas DataFrame with columns "antecedent support",
      "consequent support",
      "support", "confidence", "lift",
      "leverage", "conviction"
      of all rules for which
      metric(rule) >= min_threshold.

    """

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    def conviction_helper(sAC, sA, sC):
        confidence = sAC/sA
        conviction = np.empty(confidence.shape, dtype=float)
        conviction[:] = np.inf
        conviction[confidence < 1.] = ((1. - sC[confidence < 1.]) /
                                       (1. - confidence[confidence < 1.]))

        return conviction

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, __: sA,
        "consequent support": lambda _, __, sC: sC,
        "support": lambda sAC, _, __: sAC,
        "confidence": lambda sAC, sA, _: sAC/sA,
        "lift": lambda sAC, sA, sC: metric_dict["confidence"](sAC, sA, sC)/sC,
        "leverage": lambda sAC, sA, sC: metric_dict["support"](
             sAC, sA, sC) - sA*sC,
        "conviction": lambda sAC, sA, sC: conviction_helper(sAC, sA, sC)
        }

    columns_ordered = ["antecedent support", "consequent support",
                       "support",
                       "confidence", "lift",
                       "leverage", "conviction"]

    # check for metric compliance
    if metric not in metric_dict.keys():
        raise ValueError("Metric must be 'confidence' or 'lift', got '{}'"
                         .format(metric))

    # get dict of {frequent itemset} -> support
    keys = df['itemsets'].values
    values = df['support'].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedants = []
    rule_consequents = []
    rule_supports = []

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k)-1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)
                sA = frequent_items_dict[antecedent]
                sC = frequent_items_dict[consequent]
                # check for the threshold
                if metric_dict[metric](sAC, sA, sC) >= min_threshold:
                    rule_antecedants.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedants", "consequents"] + columns_ordered)
    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        sAC = rule_supports[0]
        sA = rule_supports[1]
        sC = rule_supports[2]
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedants, rule_consequents)),
            columns=["antecedants", "consequents"])
        for m in columns_ordered:
            df_res[m] = metric_dict[m](sAC, sA, sC)

        return df_res
