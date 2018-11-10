## bootstrap

*bootstrap(x, func, num_rounds=1000, ci=0.95, ddof=1, seed=None)*

Implements the ordinary nonparametric bootstrap

**Parameters**


- `x` : NumPy array, shape=(n_samples, [n_columns])

    An one or multidimensional array of data records


- `func` : <func>

    A function which computes a statistic that is used
    to compute the bootstrap replicates (the statistic computed
    from the bootstrap samples). This function must return a
    scalar value. For example, `np.mean` or `np.median` would be
    an acceptable argument for `func` if `x` is a 1-dimensional array
    or vector.


- `num_rounds` : int (default=1000)

    The number of bootstrap samnples to draw where each
    bootstrap sample has the same number of records as the
    original dataset.


- `ci` : int (default=0.95)

    An integer in the range (0, 1) that represents the
    confidence level for computing the confidence interval.
    For example, `ci=0.95` (default)
    will compute the 95% confidence
    interval from the bootstrap replicates.


- `ddof` : int

    The delta degrees of freedom used when computing the
    standard error.


- `seed` : int or None (default=None)

    Random seed for generating bootstrap samples.

**Returns**


- `original, standard_error, (lower_ci, upper_ci)` : tuple

    Returns the statistic of the original sample (`original`),
    the standard error of the estimate, and the
    respective confidence interval bounds.

**Examples**

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
[http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap/](http://rasbt.github.io/mlxtend/user_guide/evaluate/bootstrap/)

