# Sebastian Raschka 2014-2024
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

_metrics = [
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "representativity",
    "leverage",
    "conviction",
    "zhangs_metric",
    # "jaccard",
    # "certainty",
    # "kulczynski"
]


def association_rules(
    df: pd.DataFrame,
    df_or: pd.DataFrame,
    num_itemsets: int,
    disabled: np.ndarray,
    metric="confidence",
    min_threshold=0.8,
    support_only=False,
    return_metrics: list = _metrics,
) -> pd.DataFrame:
    """Generates a DataFrame of association rules including the
    metrics 'score', 'confidence', and 'lift'

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']

    metric : string (default: 'confidence')
      Metric to evaluate if a rule is of interest.
      **Automatically set to 'support' if `support_only=True`.**
      Otherwise, supported metrics are 'support', 'confidence', 'lift',
      'leverage', 'conviction' and 'zhangs_metric'
      These metrics are computed as follows:

      - support(A->C) = support(A+C) [aka 'support'], range: [0, 1]\n
      - confidence(A->C) = support(A+C) / support(A), range: [0, 1]\n
      - lift(A->C) = confidence(A->C) / support(C), range: [0, inf]\n
      - leverage(A->C) = support(A->C) - support(A)*support(C),
        range: [-1, 1]\n
      - conviction = [1 - support(C)] / [1 - confidence(A->C)],
        range: [0, inf]\n
      - zhangs_metric(A->C) =
        leverage(A->C) / max(support(A->C)*(1-support(A)), support(A)*(support(C)-support(A->C)))
        range: [-1,1]\n

    min_threshold : float (default: 0.8)
      Minimal threshold for the evaluation metric,
      via the `metric` parameter,
      to decide whether a candidate rule is of interest.

    support_only : bool (default: False)
      Only computes the rule support and fills the other
      metric columns with NaNs. This is useful if:

      a) the input DataFrame is incomplete, e.g., does
      not contain support values for all rule antecedents
      and consequents

      b) you simply want to speed up the computation because
      you don't need the other metrics.

    Returns
    ----------
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

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/

    """
    if not df.shape[0]:
        raise ValueError(
            "The input DataFrame `df` containing " "the frequent itemsets is empty."
        )

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError(
            "Dataframe needs to contain the\
                         columns 'support' and 'itemsets'"
        )

    def kulczynski_helper(sAC, sA, sC):
        conf_AC = sAC / sA
        conf_CA = sAC / sC
        kulczynski = (conf_AC + conf_CA) / 2
        return kulczynski

    def conviction_helper(conf, sC):
        conviction = np.empty(conf.shape, dtype=float)
        if not len(conviction.shape):
            conviction = conviction[np.newaxis]
            conf = conf[np.newaxis]
            sC = sC[np.newaxis]
        conviction[:] = np.inf
        conviction[conf < 1.0] = (1.0 - sC[conf < 1.0]) / (
            1.0 - conf[conf < 1.0]
        )

        return conviction

    def zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_):
        denominator = np.maximum(sAC * (1 - sA), sA * (sC - sAC))
        numerator = metric_dict["leverage"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)

        with np.errstate(divide="ignore", invalid="ignore"):
            # ignoring the divide by 0 warning since it is addressed in the below np.where
            zhangs_metric = np.where(denominator == 0, 0, numerator / denominator)

        return zhangs_metric

    def jaccard_metric_helper(sAC, sA, sC):
        numerator = metric_dict["support"](sAC, sA, sC)
        denominator = sA + sC - numerator

        jaccard_metric = numerator / denominator
        return jaccard_metric

    def certainty_metric_helper(sAC, sA, sC):
        certainty_num = metric_dict["confidence"](sAC, sA, sC) - sC
        certainty_denom = 1 - sC

        cert_metric = np.where(certainty_denom == 0, 0, certainty_num / certainty_denom)
        return cert_metric

    # metrics for association rules
    metric_dict = {
        "antecedent support": lambda _, sA, ___, ____, _____, ______, _______, ________: sA,
        "consequent support": lambda _, __, sC, ____, _____, ______, _______, ________: sC,
        "support": lambda sAC, _, __, ___, ____, _____, ______, _______: sAC,
        "confidence": lambda sAC, sA, _, disAC, disA, __, dis_int, ___: (sAC*(num_itemsets - disAC)) / (sA*(num_itemsets - disA) - dis_int),
        "lift": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: metric_dict["confidence"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_) / (sC*(num_itemsets - disC) - dis_int_),
        "representativity": lambda _, __, disAC, ____, ___, ______, _______, ________ : (num_itemsets-disAC)/num_itemsets,
        "leverage": lambda sAC, sA, sC, _, __, ____, _____, ______: metric_dict["support"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_) - sA * sC,
        "conviction": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: conviction_helper(metric_dict["confidence"](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_), sC),
        "zhangs_metric": lambda sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_: zhangs_metric_helper(sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_),
        # "jaccard": lambda sAC, sA, sC: jaccard_metric_helper(sAC, sA, sC),
        # "certainty": lambda sAC, sA, sC: certainty_metric_helper(sAC, sA, sC),
        # "kulczynski": lambda sAC, sA, sC: kulczynski_helper(sAC, sA, sC),
    }

    # check for metric compliance
    if support_only:
        metric = "support"
    else:
        if metric not in metric_dict.keys():
            raise ValueError(
                "Metric must be 'confidence' or 'lift', got '{}'".format(metric)
            )

    # get dict of {frequent itemset} -> support
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    # prepare buckets to collect frequent rules
    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    # assign columns from original df to be the same on the disabled.
    disabled = pd.DataFrame(disabled)
    disabled.columns = df_or.columns

    # iterate over all frequent itemsets
    for k in frequent_items_dict.keys():
        sAC = frequent_items_dict[k]
        # to find all possible combinations
        for idx in range(len(k) - 1, 0, -1):
            # of antecedent and consequent
            for c in combinations(k, r=idx):
                antecedent = frozenset(c)
                consequent = k.difference(antecedent)

                if support_only:
                    # support doesn't need these,
                    # hence, placeholders should suffice
                    sA = None
                    sC = None

                else:
                    try:
                        sA = frequent_items_dict[antecedent]
                        sC = frequent_items_dict[consequent]

                        an=list(antecedent)
                        con=list(consequent)
                        an.extend(con)

                        dec=disabled.loc[:,an]
                        _dec=disabled.loc[:,list(antecedent)]
                        __dec=disabled.loc[:,list(consequent)]
                        dec_=df_or.loc[:,list(antecedent)]
                        dec__=df_or.loc[:,list(consequent)]

                        disAC=0
                        disA=0
                        disC=0
                        dis_int=0
                        dis_int_=0
                        for i in range(len(dec.index)):
                            v=list(dec.iloc[i,:])
                            x=list(_dec.iloc[i,:])
                            y=list(__dec.iloc[i,:])
                            z=list(dec_.iloc[i,:])
                            w=list(dec__.iloc[i,:])
                            # if (1 in x) and all(x=='?' for x in y): ##
                            if 1 in set(v):
                                disAC+=1
                            if 1 in set(x):
                                disA+=1
                            if 1 in y:
                                disC+=1

                        # for i in range(len(_dec.index)):
                            # x=list(_dec.iloc[i,:])
                            # y=list(__dec.iloc[i,:])
                            # if 1 in set(x):
                            #     disA+=1
                            # if 1 in y:
                            #     disC+=1

                        # for i in range(len(__dec.index)):
                        #     x=list(__dec.iloc[i,:])
                        #     y=list(_dec.iloc[i,:])
                        #     z=list(dec_.iloc[i,:])
                        #     w=list(dec__.iloc[i,:])
                            # if (1 in x) and (True in np.isnan(y)) and (0 not in z):

                            # if (1 in x) and ((0 not in z) and ('?' not in z)):
                            # if (1 in x) and ((0 not in z) and (1 in z)):
                            if (1 in y) and all(j==1 for j in z):
                                dis_int+=1
                            # if (1 in y) and ((0 not in w) and ('?' not in w)):
                            if (1 in x) and all(j==1 for j in w):
                                dis_int_+=1

                    except KeyError as e:
                        s = (
                            str(e) + "You are likely getting this error"
                            " because the DataFrame is missing "
                            " antecedent and/or consequent "
                            " information."
                            " You can try using the "
                            " `support_only=True` option"
                        )
                        raise KeyError(s)
                    # check for the threshold

                score = metric_dict[metric](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)
                # print(score)
                if score >= min_threshold:
                    rule_antecedents.append(antecedent)
                    rule_consequents.append(consequent)
                    rule_supports.append([sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_])

    # check if frequent rule was generated
    if not rule_supports:
        return pd.DataFrame(columns=["antecedents", "consequents"] + return_metrics)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["antecedents", "consequents"],
        )

        if support_only:
            sAC = rule_supports[0]
            for m in return_metrics:
                df_res[m] = np.nan
            df_res["support"] = sAC

        else:
            sAC = rule_supports[0]
            sA = rule_supports[1]
            sC = rule_supports[2]
            disAC = rule_supports[3]
            disA = rule_supports[4]
            disC = rule_supports[5]
            dis_int = rule_supports[6]
            dis_int_= rule_supports[7]
            
            for m in return_metrics:
                df_res[m] = metric_dict[m](sAC, sA, sC, disAC, disA, disC, dis_int, dis_int_)

        return df_res
