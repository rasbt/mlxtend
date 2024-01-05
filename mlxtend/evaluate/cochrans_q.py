# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import itertools

import numpy as np
import scipy.stats


def cochrans_q(y_target, *y_model_predictions):
    """
    Cochran's Q test to compare 2 or more models.

    Parameters
    -----------
    y_target : array-like, shape=[n_samples]
        True class labels as 1D NumPy array.

    *y_model_predictions : array-likes, shape=[n_samples]
        Variable number of 2 or more arrays that
        contain the predicted class labels
        from models as 1D NumPy array.

    Returns
    -----------

    q, p : float or None, float
        Returns the Q (chi-squared) value and the p-value

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/

    """

    num_models = len(y_model_predictions)

    # Checks
    model_lens = set()
    y_model_predictions = list(y_model_predictions)
    for ary in [y_target] + y_model_predictions:
        if len(ary.shape) != 1:
            raise ValueError("One or more input arrays are not 1-dimensional.")
        model_lens.add(ary.shape[0])

    if len(model_lens) > 1:
        raise ValueError(
            "Each prediction array must have the " "same number of samples."
        )

    if num_models < 2:
        raise ValueError("Provide at least 2 model prediction arrays.")

    # Q test statistic
    degrees_of_freedom = num_models - 1

    # numerator
    correctly_classified_all_models = 0
    correctly_classified_collection = []
    for pred in y_model_predictions:
        correctly_classified = (y_target == pred).sum()
        correctly_classified_all_models += correctly_classified
        correctly_classified_collection.append(correctly_classified)

    numerator = (
        num_models * sum([c**2 for c in correctly_classified_collection])
        - correctly_classified_all_models**2
    )

    # denominator
    binary_combin = list(itertools.product([0, 1], repeat=num_models))
    ary = np.hstack(
        [(y_target == mod).reshape(-1, 1) for mod in y_model_predictions]
    ).astype(int)
    correctly_classified_objects = 0
    binary_combin_totals = np.zeros(len(binary_combin))
    for i, c in enumerate(binary_combin):
        binary_combin_totals[i] = ((ary == c).sum(axis=1) == num_models).sum()

        correctly_classified_objects += sum(c) ** 2 * binary_combin_totals[i]

    denominator = (
        num_models * correctly_classified_all_models - correctly_classified_objects
    )

    q = degrees_of_freedom * numerator / denominator
    p_value = scipy.stats.chi2.sf(q, degrees_of_freedom)

    return q, p_value
