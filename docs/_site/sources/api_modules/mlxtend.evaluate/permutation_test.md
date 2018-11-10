## permutation_test

*permutation_test(x, y, func='x_mean != y_mean', method='exact', num_rounds=1000, seed=None)*

Nonparametric permutation test

**Parameters**

- `x` : list or numpy array with shape (n_datapoints,)

    A list or 1D numpy array of the first sample
    (e.g., the treatment group).

- `y` : list or numpy array with shape (n_datapoints,)

    A list or 1D numpy array of the second sample
    (e.g., the control group).

- `func` : custom function or str (default: 'x_mean != y_mean')

    function to compute the statistic for the permutation test.
    - If 'x_mean != y_mean', uses
    `func=lambda x, y: np.abs(np.mean(x) - np.mean(y)))`
    for a two-sided test.
    - If 'x_mean > y_mean', uses
    `func=lambda x, y: np.mean(x) - np.mean(y))`
    for a one-sided test.
    - If 'x_mean < y_mean', uses
    `func=lambda x, y: np.mean(y) - np.mean(x))`
    for a one-sided test.

- `method` : 'approximate' or 'exact' (default: 'exact')

    If 'exact' (default), all possible permutations are considered.
    If 'approximate' the number of drawn samples is
    given by `num_rounds`.
    Note that 'exact' is typically not feasible unless the dataset
    size is relatively small.

- `num_rounds` : int (default: 1000)

    The number of permutation samples if `method='approximate'`.

- `seed` : int or None (default: None)

    The random seed for generating permutation samples if
    `method='approximate'`.

**Returns**

p-value under the null hypothesis

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/](http://rasbt.github.io/mlxtend/user_guide/evaluate/permutation_test/)

