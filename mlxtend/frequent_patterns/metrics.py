# mlxtend Machine Learning Library Extensions
#
# Functions to measure quality of association rules
#
# Author: Mohammed Niyas <https://github.com/mniyas>
#
# License: BSD 3 clause


def kulczynski_measure(df, antecedent, consequent):
    """Calculates the Kulczynski measure for a given rule.

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of association rules
      with columns ['antecedents', 'consequents', 'confidence']

    antecedent : set or frozenset
        Antecedent of the rule
    consequent : set or frozenset
        Consequent of the rule

    Returns
    ----------
    The Kulczynski measure
        K(A,C) = (confidence(A->C) + confidence(C->A)) / 2, range: [0, 1]\n.
    """
    if not df.shape[0]:
        raise ValueError('The input DataFrame `df` containing '
                         'the frequent itemsets is empty.')

    # check for mandatory columns
    required_columns = ["antecedents", "consequents", "confidence"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(
            "Dataframe needs to contain the\
            columns 'antecedents', 'consequents' and 'confidence'"
        )

    # get confidence of antecedent to consequent rule
    a_to_c = df[
        (df["antecedents"] == antecedent) & (df["consequents"] == consequent)]
    try:
        a_to_c_confidence = a_to_c["confidence"].iloc[0]
    except IndexError:
        a_to_c_confidence = 0

    # get confidence of consequent to antecedent rule
    c_to_a = df[
        (df["antecedents"] == consequent) & (df["consequents"] == antecedent)]
    try:
        c_to_a_confidence = c_to_a["confidence"].iloc[0]
    except IndexError:
        c_to_a_confidence = 0
    return (a_to_c_confidence + c_to_a_confidence) / 2


def imbalance_ratio(df, a, b):
    """
    Calculates the imbalance ratio for a given pair of itemsets

    Parameters
    -----------
    df : pandas DataFrame
      pandas DataFrame of frequent itemsets
      with columns ['support', 'itemsets']
    a : set or frozenset
        First itemset
    b : set or frozenset
        Second itemset

    Returns
    ----------
    The imbalance ratio
         I(A,B) = |support(A) - support(B)| /\
            (support(A) + support(B) - support(A+B)), range: [0, 1]\n.
    """
    if not df.shape[0]:
        raise ValueError('The input DataFrame `df` containing '
                         'the frequent itemsets is empty.')

    # check for mandatory columns
    if not all(col in df.columns for col in ["support", "itemsets"]):
        raise ValueError("Dataframe needs to contain the\
                         columns 'support' and 'itemsets'")

    # get support of a
    try:
        sA = df[df["itemsets"] == a].support.iloc[0]
    except IndexError:
        sA = 0

    # get support of b
    try:
        sB = df[df["itemsets"] == b].support.iloc[0]
    except IndexError:
        sB = 0

    # get support of a union b
    try:
        sAB = df[df["itemsets"] == a.union(b)].support.iloc[0]
    except IndexError:
        sAB = 0

    try:
        return abs(sA - sB) / (sA + sB - sAB)
    except ZeroDivisionError:
        return 0
