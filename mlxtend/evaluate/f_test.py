# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import itertools

import numpy as np
import scipy.stats
from sklearn.metrics import get_scorer
from sklearn.model_selection import train_test_split


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
    https://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/

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
    ssa = (
        num_examples * sum([acc**2 for acc in accuracies])
        - num_examples * num_models * avg_acc**2
    )

    # sum squares of models
    binary_combin = list(itertools.product([0, 1], repeat=num_models))
    ary = np.hstack(
        [(y_target == mod).reshape(-1, 1) for mod in y_model_predictions]
    ).astype(int)
    correctly_classified_objects = 0
    binary_combin_totals = np.zeros(len(binary_combin))
    for i, c in enumerate(binary_combin):
        binary_combin_totals[i] = ((ary == c).sum(axis=1) == num_models).sum()

        correctly_classified_objects += sum(c) ** 2 * binary_combin_totals[i]

    ssb = (
        1.0 / num_models * correctly_classified_objects
        - num_examples * num_models * avg_acc**2
    )

    # total sum of squares
    sst = num_examples * num_models * avg_acc * (1 - avg_acc)

    # sum squares for classification-object interaction
    ssab = sst - ssa - ssb

    mean_ssa = ssa / (num_models - 1)
    mean_ssab = ssab / ((num_models - 1) * (num_examples - 1))

    f = mean_ssa / mean_ssab

    degrees_of_freedom_1 = num_models - 1
    degrees_of_freedom_2 = degrees_of_freedom_1 * num_examples

    p_value = scipy.stats.f.sf(f, degrees_of_freedom_1, degrees_of_freedom_2)

    return f, p_value


def combined_ftest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None):
    """
    Implements the 5x2cv combined F test proposed
    by Alpaydin 1999,
    to compare the performance of two models.

    Parameters
    ----------
    estimator1 : scikit-learn classifier or regressor

    estimator2 : scikit-learn classifier or regressor

    X : {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape = [n_samples]
        Target values.

    scoring : str, callable, or None (default: None)
        If None (default), uses 'accuracy' for sklearn classifiers
        and 'r2' for sklearn regressors.
        If str, uses a sklearn scoring metric string identifier, for example
        {accuracy, f1, precision, recall, roc_auc} for classifiers,
        {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
        'median_absolute_error', 'r2'} for regressors.
        If a callable object or function is provided, it has to be conform with
        sklearn's signature ``scorer(estimator, X, y)``; see
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
        for more information.

    random_seed : int or None (default: None)
        Random seed for creating the test/train splits.

    Returns
    ----------
    f : float
        The F-statistic

    pvalue : float
        Two-tailed p-value.
        If the chosen significance level is larger
        than the p-value, we reject the null hypothesis
        and accept that there are significant differences
        in the two compared models.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/evaluate/combined_ftest_5x2cv/

    """
    rng = np.random.RandomState(random_seed)

    if scoring is None:
        if estimator1._estimator_type == "classifier":
            scoring = "accuracy"
        elif estimator1._estimator_type == "regressor":
            scoring = "r2"
        else:
            raise AttributeError("Estimator must " "be a Classifier or Regressor.")
    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    variances = []
    differences = []

    def score_diff(X_1, X_2, y_1, y_2):
        estimator1.fit(X_1, y_1)
        estimator2.fit(X_1, y_1)
        est1_score = scorer(estimator1, X_2, y_2)
        est2_score = scorer(estimator2, X_2, y_2)
        score_diff = est1_score - est2_score
        return score_diff

    for i in range(5):
        randint = rng.randint(low=0, high=32767)
        X_1, X_2, y_1, y_2 = train_test_split(X, y, test_size=0.5, random_state=randint)

        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2

        differences.extend([score_diff_1**2, score_diff_2**2])
        variances.append(score_var)

    numerator = sum(differences)
    denominator = 2 * (sum(variances))
    f_stat = numerator / denominator

    p_value = scipy.stats.f.sf(f_stat, 10, 5)

    return float(f_stat), float(p_value)
