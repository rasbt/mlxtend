# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# Bootstrap functions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def bootstrap(x, func, num_rounds=1000, ci=0.95, ddof=1, seed=None):
    """Implements the ordinary nonparametric bootstrap

    Parameters
    ----------

    x : NumPy array, shape=(n_samples, [n_columns])
        An one or multidimensional array of data records

    func : <func>
        A function which computes a statistic that is used
        to compute the bootstrap replicates (the statistic computed
        from the bootstrap samples). This function must return a
        scalar value. For example, `np.mean` or `np.median` would be
        an acceptable argument for `func` if `x` is a 1-dimensional array
        or vector.

    num_rounds : int (default=1000)
        The number of bootstrap samnples to draw where each
        bootstrap sample has the same number of records as the
        original dataset.

    ci : int (default=0.95)
        An integer in the range (0, 1) that represents the
        confidence level for computing the confidence interval.
        For example, `ci=0.95` (default)
        will compute the 95% confidence
        interval from the bootstrap replicates.

    ddof : int
        The delta degrees of freedom used when computing the
        standard error.

    seed : int or None (default=None)
        Random seed for generating bootstrap samples.

    Returns
    -------

    original, standard_error, (lower_ci, upper_ci) : tuple
        Returns the statistic of the original sample (`original`),
        the standard error of the estimate, and the
        respective confidence interval bounds.

    Examples
    --------

    >>> from mlxtend.evaluate import bootstrap
    >>> rng = np.random.RandomState(123)
    >>> x = rng.normal(loc=5., size=100)
    >>> original, std_err, ci_bounds = bootstrap(x,
    ...                                          num_rounds=1000,
    ...                                          func=np.mean,
    ...                                          ci=0.95,
    ...                                          seed=123)
    >>> print('Mean: %.2f, SE: +/- %.2f, CI95: [%.2f, %.2f]' % (original,
    ...                                                         std_err,
    ...                                                         ci_bounds[0],
    ...                                                         ci_bounds[1]))
    Mean: 5.03, SE: +/- 0.11, CI95: [4.80, 5.26]
    >>>

    For more usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap/

    """
    if ci <= 0 or ci >= 1:
        raise AttributeError('ci must be in range (0, 1)')

    check_output = func(x)

    if (not isinstance(check_output, float)
            and not isinstance(check_output, int)
            and len(check_output.shape) != 0):
        raise AttributeError('func must return a scalar')

    rng = np.random.RandomState(seed)
    bootstrap_replicates = np.zeros(shape=num_rounds)

    # quantile function implemented due
    # to the weird behavior of the NumPy equivalent with
    # either nearest or lower interpolation
    def quantile(x, q):
        rank = round(q * x.shape[0]) - 1
        if rank >= x.shape[0]:
            rank = x.shape[0]
        elif rank <= 0:
            rank = 0
        rank = int(round(rank))
        return x[rank]

    sample_idx = np.arange(x.shape[0])
    for i in range(num_rounds):
        bootstrap_idx = rng.choice(sample_idx,
                                   size=sample_idx.shape[0],
                                   replace=True)

        bootstrap_replicates[i] = func(x[bootstrap_idx])

    original = check_output
    standard_error = np.std(bootstrap_replicates, ddof=ddof)

    t = np.sort(bootstrap_replicates)
    bound = (1 - ci) / 2.

    upper_ci = quantile(t, q=(ci + bound))
    lower_ci = quantile(t, q=bound)

    return original, standard_error, (lower_ci, upper_ci)
