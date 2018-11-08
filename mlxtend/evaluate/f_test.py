# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import scipy.stats
import itertools


def ftest(y_target, *y_model_predictions):
    """
    F-Test test to compare 2 or more models.

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

    f, p : float or None, float
        Returns the F-value and the p-value

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/

    """

    num_models = len(y_model_predictions)

    # Checks
    model_lens = set()
    y_model_predictions = list(y_model_predictions)
    for ary in ([y_target] + y_model_predictions):
        if len(ary.shape) != 1:
            raise ValueError('One or more input arrays are not 1-dimensional.')
        model_lens.add(ary.shape[0])

    if len(model_lens) > 1:
        raise ValueError('Each prediction array must have the '
                         'same number of samples.')

    if num_models < 2:
        raise ValueError('Provide at least 2 model prediction arrays.')

    num_examples = len(y_target)

    accuracies = []
    correctly_classified_all_models = 0
    correctly_classified_collection = []
    for pred in y_model_predictions:
        correctly_classified = (y_target == pred).sum()
        acc = correctly_classified / num_examples
        accuracies.append(acc)
        correctly_classified_all_models += correctly_classified
        correctly_classified_collection.append(correctly_classified)

    avg_acc = sum(accuracies) / len(accuracies)

    # sum squares of classifiers
    ssa = (num_examples * sum([acc**2 for acc in accuracies])
           - num_examples*num_models*avg_acc**2)

    # sum squares of models
    binary_combin = list(itertools.product([0, 1], repeat=num_models))
    ary = np.hstack(((y_target == mod).reshape(-1, 1) for
                    mod in y_model_predictions)).astype(int)
    correctly_classified_objects = 0
    binary_combin_totals = np.zeros(len(binary_combin))
    for i, c in enumerate(binary_combin):
        binary_combin_totals[i] = ((ary == c).sum(axis=1) == num_models).sum()

        correctly_classified_objects += (sum(c)**2 * binary_combin_totals[i])

    ssb = (1./num_models * correctly_classified_objects
           - num_examples*num_models*avg_acc**2)

    # total sum of squares
    sst = num_examples*num_models*avg_acc*(1 - avg_acc)

    # sum squares for classification-object interaction
    ssab = sst - ssa - ssb

    mean_ssa = ssa / (num_models - 1)
    mean_ssab = ssab / ((num_models - 1)*(num_examples - 1))

    f = mean_ssa / mean_ssab

    degrees_of_freedom_1 = num_models - 1
    degrees_of_freedom_2 = degrees_of_freedom_1 * num_examples

    p_value = scipy.stats.f.sf(f, degrees_of_freedom_1, degrees_of_freedom_2)

    return f, p_value
