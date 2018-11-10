# Bootstrap

An implementation of the ordinary nonparametric bootstrap to bootstrap a single statistic (for example, the mean. median, R^2 of a regression fit, and so forth).

> `from mlxtend.evaluate import bootstrap`    

## Overview

The bootstrap offers an easy and effective way to estimate the distribution of a statistic via simulation, by drawing (or generating) new samples from an existing sample with replacement. Note that the bootstrap does not require making any assumptions about the sample statistic or dataset being normally distributed.

Using the bootstrap, we can estimate sample statistics and compute the standard error of the mean and confidence intervals as if we have drawn a number of samples from an infinite population. In a nutshell, the bootstrap procedure can be described as follows:

1. Draw a sample with replacement
2. Compute the sample statistic
3. Repeat step 1-2 n times
4. Compute the standard deviation (standard error of the mean of the statistic)
5. Compute the confidence interval

Or, in simple terms, we can interpret the bootstrap a means of drawing a potentially endless number of (new) samples from a population by resampling the original dataset. 

Note that the term "bootstrap replicate" is being used quite loosely in current literature; many researchers and practitioners use it to define the number of bootstrap samples we draw from the original dataset. However, in the context of this documentation and the code annotation, we use the original definition of bootstrap repliactes and use it to refer to the statistic computed from a bootstrap sample.

### References

- [1]  Efron, Bradley, and Robert J. Tibshirani. An introduction to the bootstrap. CRC press, 1994. Management of Data (ACM SIGMOD '97), pages 265-276, 1997.

## Example 1 -- Bootstrapping the Mean

This simple example illustrates how you could bootstrap the mean of a sample.


```python
import numpy as np
from mlxtend.evaluate import bootstrap


rng = np.random.RandomState(123)
x = rng.normal(loc=5., size=100)
original, std_err, ci_bounds = bootstrap(x, num_rounds=1000, func=np.mean, ci=0.95, seed=123)
print('Mean: %.2f, SE: +/- %.2f, CI95: [%.2f, %.2f]' % (original, 
                                                             std_err, 
                                                             ci_bounds[0],
                                                             ci_bounds[1]))
```

    Mean: 5.03, SE: +/- 0.11, CI95: [4.80, 5.26]


## Example 2 - Bootstrapping a Regression Fit

This example illustrates how you can bootstrap the $R^2$ of a regression fit on the training data.


```python
from mlxtend.data import autompg_data

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X, y = autompg_data()


lr = LinearRegression()

def r2_fit(X, model=lr):
    x, y = X[:, 0].reshape(-1, 1), X[:, 1]
    pred = lr.fit(x, y).predict(x)
    return r2_score(y, pred)
    
    
original, std_err, ci_bounds = bootstrap(X, num_rounds=1000,
                                         func=r2_fit,
                                         ci=0.95,
                                         seed=123)
print('Mean: %.2f, SE: +/- %.2f, CI95: [%.2f, %.2f]' % (original, 
                                                             std_err, 
                                                             ci_bounds[0],
                                                             ci_bounds[1]))
```

    Mean: 0.90, SE: +/- 0.01, CI95: [0.89, 0.92]


## API


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


