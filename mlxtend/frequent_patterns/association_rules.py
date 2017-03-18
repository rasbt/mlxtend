# Sebastian Raschka 2014-2017
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
      Supported metrics are 'confidence' and 'lift'
    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric
      to decide whether a candidate rule is of interest.

    Returns
    ----------
    pandas DataFrame with columns ['antecedants', 'consequents',
    'support', 'lift', 'confidence'] of all rules for which
    metric(rule) >= min_threshold.

    """

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    # metrics for association rules
    metric_dict = {
        "confidence": lambda sXY, sX, _:
        sXY/sX,
        "lift": lambda sXY, sX, sY:
        metric_dict["confidence"](sXY, sX, sY)/sY
        }

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
        sXY = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k)-1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)
                sX = frequent_items_dict[antecedent]
                sY = frequent_items_dict[consequent]
                # check for the threshold
                if metric_dict[metric](sXY, sX, sY) >= min_threshold:
                    rule_antecedants.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sXY, sX, sY])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(
            columns=["antecedants", "consequents", "support"]
            .append(list(metric_dict.keys())))
    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T
        sXY = rule_supports[0]
        sX = rule_supports[1]
        sY = rule_supports[2]
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedants, rule_consequents, sX)),
            columns=["antecedants", "consequents", "support"])
        for m in sorted(metric_dict.keys()):
            df_res[m] = metric_dict[m](sXY, sX, sY)

        return df_res
