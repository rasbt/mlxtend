# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
#
# A function for plotting learning curves of classifiers.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_learning_curves(X_train, y_train,
                         X_test, y_test,
                         estimator,
                         train_marker='o',
                         test_marker='^',
                         scoring='misclassification error',
                         suppress_plot=False,
                         print_model=True,
                         title_fontsize=12,
                         style='default',
                         legend_loc='best'):
    """Plots learning curves of a classifier or regressor.

    Parameters
    ----------
    X_train : array-like, shape = [n_samples, n_features]
        Feature matrix of the training dataset.
    y_train : array-like, shape = [n_samples]
        True class labels of the training dataset.
    X_test : array-like, shape = [n_samples, n_features]
        Feature matrix of the test dataset.
    y_test : array-like, shape = [n_samples]
        True class labels of the test dataset.
    estimator : Classifier object. Must have a .predict .fit method.
    train_marker : str (default: 'o')
        Marker for the training set line plot.
    test_marker : str (default: '^')
        Marker for the test set line plot.
    scoring : str, callable or None (default="misclassification error")
        Scoring-function based on scikit-learn:
        - None: Uses the estimator's default scoring function 
        - String (see http://scikit-learn.org/stable/modules/
                  model_evaluation.html#common-cases-predefined-values)
        - 'misclassification error': 1-accuracy
        - A scorer, callable object / function 
        with signature `scorer(estimator, X, y)`.
    suppress_plot=False : bool (default: False)
        Suppress matplotlib plots if True. Recommended
        for testing purposes.
    print_model : bool (default: True)
        Print model parameters in plot title if True.
    title_fontsize : int (default: 12)
        Determines the size of the plot title font.
    style : str (default: 'default')
        Matplotlib style. For more styles, please see
        https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    legend_loc : str (default: 'best')
        Where to place the plot legend:
        {'best', 'upper left', 'upper right', 'lower left', 'lower right'}

    Returns
    ---------
    errors : (training_error, test_error): tuple of lists

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/plotting/plot_learning_curves/

    """
    if scoring != 'misclassification error':

        scorer = metrics.scorer.check_scoring(
                                    estimator=estimator,
                                    scoring=scoring,
                                    )
 
    else:

        def complement_scorer(scorer):
            """Returns a scorer that computes the complement of the scorer argument
            in the sense of 1 - scorer(*args)
        
            Parameters
            ----------
            scorer : Any callable that evaluates to a float or int between 0 and 1, e.g.
            accuracy scorer. Required signature for scorer is (estimator, X,
            y_true), keyword args optional (scorer will not be invoked with keyword
            args; in particular sample_weights is not currently supported).
        
            Returns
            ---------
            complemented_scorer : callable
        
            """
            def complemented_scorer(estimator, X, y_true):
                return 1 - scorer(estimator, X, y_true)

            return complemented_scorer


        accuracy_scorer = metrics.scorer.check_scoring(
                                    estimator=estimator,
                                    scoring="accuracy",
                                    )

        scorer = complement_scorer(accuracy_scorer)
    
    training_errors = []
    test_errors = []

    rng = [int(i) for i in np.linspace(0, X_train.shape[0], 11)][1:]
    for r in rng:

        model = estimator.fit(X_train[:r], y_train[:r])

        train_misclf = scorer(model, X_train[:r], y_train[:r])
        training_errors.append(train_misclf)

        test_misclf = scorer(model, X_test, y_test)
        test_errors.append(test_misclf)

    if not suppress_plot:
        with plt.style.context(style):
            plt.plot(np.arange(10, 101, 10), training_errors,
                     label='training set', marker=train_marker)
            plt.plot(np.arange(10, 101, 10), test_errors,
                     label='test set', marker=test_marker)
            plt.xlabel('Training set size in percent')

    if not suppress_plot:
        with plt.style.context(style):
            plt.ylabel('Performance ({})'.format(scoring))
            if print_model:
                plt.title(
                    'Learning Curves\n\n{}\n'.format(model),
                    fontsize=title_fontsize)
            plt.legend(loc=legend_loc, numpoints=1)
            plt.xlim([0, 110])
            max_y = max(max(test_errors), max(training_errors))
            min_y = min(min(test_errors), min(training_errors))
            plt.ylim([min_y - min_y * 0.15, max_y + max_y * 0.15])
    errors = (training_errors, test_errors)
    return errors
