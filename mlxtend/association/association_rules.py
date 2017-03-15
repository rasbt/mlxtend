def generate_rules(df, metric="confidence", min_threshold=0.8):
    """"""
    # Metrics for association rules
    metric_dict = {
        "confidence": lambda sXY, sX, _:  sXY/sX,
        "lift": lambda sXY, sX, sY: metric_dict["confidence"](sXY, sX, sY)/sY,
        "conviction": lambda sXY, sX, sY: \
        float("inf") if  metric_dict["confidence"](sXY, sX, sY) == 1 \
        else (1-sY) / (1-metric_dict["confidence"](sXY, sX, sY))
        }
    
    # check for metric compliance
    if metric not in metric_dict.keys():
        pass # raise error
    
    # get dict of {frequent itemset} -> support
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    keys = df.values.T[1]
    values = df.values.T[0]
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
        return
    else:
        # Change conviction calculation to support broadcasting
        metric_dict["conviction"] = lambda sXY, sX, sY: \
        (1-sY) / (1-metric_dict["confidence"](sXY, sX, sY))
        
        # generate ALL the metrics
        rule_supports = np.array(rule_supports).T
        sXY = rule_supports[0]
        sX = rule_supports[1]
        sY = rule_supports[2]
        df_res = pd.DataFrame(data = list(zip(rule_antecedants, rule_consequents, sX)), 
                              columns = ["antecedants", "consequents", "support"])
        for m in metric_dict.keys():
            df_res[m] = metric_dict[m](sXY, sX, sY)
        
        return df_res.replace(np.nan, np.inf)