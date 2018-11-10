# Permutation Test

An implementation of a permutation test for hypothesis testing -- testing the null hypothesis that two different groups come from the same distribution.

> `from mlxtend.evaluate import permutation_test`    

## Overview

Permutation tests (also called exact tests, randomization tests, or re-randomization tests) are nonparametric test procedures to test the null hypothesis that two different groups come from the same distribution. A permutation test can be used for significance or hypothesis testing (including A/B testing) without requiring to make any assumptions about the sampling distribution (e.g., it doesn't require the samples to be normal distributed).


Under the null hypothesis (treatment = control), any permutations are equally likely. (Note that there are (n+m)! permutations, where *n* is the number of records in the treatment sample, and *m* is the number of records in the control sample). For a two-sided test, we define the alternative hypothesis that the two samples are different (e.g., treatment != control). 

1. Compute the difference (here: mean) of sample x and sample y
2. Combine all measurements into a single dataset
3. Draw a permuted dataset from all possible permutations of the dataset in 2.
4. Divide the permuted dataset into two datasets x' and y' of size *n* and *m*, respectively
5. Compute the difference (here: mean) of sample x' and sample y' and record this difference
6. Repeat steps 3-5 until all permutations are evaluated
7. Return the p-value as the number of times the recorded differences were more extreme than the original difference from 1. and divide this number by the total number of permutations

Here, the p-value is defined as the probability, given the null hypothesis (no difference between the samples) is true, that we obtain results that are at least as extreme as the results we observed (i.e., the sample difference from 1.).

More formally, we can express the computation of the p-value as follows ([2]):

$$p(t > t_0) = \frac{1}{(n+m)!} \sum^{(n+m)!}_{j=1} I(t_j > t_0),$$

where $t_0$ is the observed value of the test statistic (1. in the list above), and $t$ is the t-value, the statistic computed from the resamples (5.) $t(x'_1, x'_2, ..., x'_n, y'_1, y'_2, ..., x'_m) = |\bar{x'} - \bar{y'}|$, and *I* is the indicator function.


Given a significance level that we specify prior to carrying out the permutation test (e.g., alpha=0.05), we fail to reject the null hypothesis if the p-value is greater than alpha.

Note that if the number of permutation is large, sampling all permutation may not computationally be feasible. Thus, a common approximation is to perfom *k* rounds of permutations (where *k* is typically a value between 1000 and 2000).

### References

- [1]  Efron, Bradley and Tibshirani, R. J., An introduction to the bootstrap, Chapman & Hall/CRC Monographs on Statistics & Applied Probability, 1994.
- [2] Unpingco, José. Python for probability, statistics, and machine learning. Springer, 2016.
- [3] Pitman, E. J. G., Significance tests which may be applied to samples from any population, Royal Statistical Society Supplement, 1937, 4: 119-30 and 225-32

## Example 1 -- Two-sided permutation test

Perform a two-sided permutation test to test the null hypothesis that two groups, "treatment" and "control" come from the same distribution. We specify alpha=0.01 as our significance level.


```python
treatment = [ 28.44,  29.32,  31.22,  29.58,  30.34,  28.76,  29.21,  30.4 ,
              31.12,  31.78,  27.58,  31.57,  30.73,  30.43,  30.31,  30.32,
              29.18,  29.52,  29.22,  30.56]
control = [ 33.51,  30.63,  32.38,  32.52,  29.41,  30.93,  49.78,  28.96,
            35.77,  31.42,  30.76,  30.6 ,  23.64,  30.54,  47.78,  31.98,
            34.52,  32.42,  31.32,  40.72]
```

Since evaluating all possible permutations may take a while, we will use the approximation method (see the introduction for details):


```python
from mlxtend.evaluate import permutation_test

p_value = permutation_test(treatment, control,
                           method='approximate',
                           num_rounds=10000,
                           seed=0)
print(p_value)
```

    0.0066


Since p-value < alpha, we can reject the null hypothesis that the two samples come from the same distribution.

## Example 2 -- Calculating the p-value for correlation analysis (Pearson's R)

Note: this is a one-sided hypothesis testing as we conduct the permutation test as "how many times obtain a correlation coefficient that is greater than the observed value?"


```python
import numpy as np
from mlxtend.evaluate import permutation_test

x = np.array([1, 2, 3, 4, 5, 6])
y = np.array([2, 4, 1, 5, 6, 7])

print('Observed pearson R: %.2f' % np.corrcoef(x, y)[1][0])


p_value = permutation_test(x, y,
                           method='exact',
                           func=lambda x, y: np.corrcoef(x, y)[1][0],
                           seed=0)
print('P value: %.2f' % p_value)
```

    Observed pearson R: 0.81
    P value: 0.09


## API


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


