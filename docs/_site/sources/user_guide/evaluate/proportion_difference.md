# Proportion Difference Test

Test of the difference of proportions for classifier performance comparison.

> `from mlxtend.evaluate import proportion_difference`    

## Overview

There are several different statistical hypothesis testing frameworks that are being used in practice to compare the performance of classification models, including common methods such as difference of two proportions (here, the proportions are the estimated generalization accuracies from a test set), for which we can construct 95% confidence intervals based on the concept of the Normal Approximation to the Binomial that was covered in Part I. 

Performing a z-score test for two population proportions is inarguably the most straight-forward way to compare to models (but certainly not the best!): In a nutshell, if the 95% confidence intervals of the accuracies of two models do not overlap, we can reject the null hypothesis that the performance of both classifiers is equal at a confidence level of $\alpha=0.05$ (or 5% probability). Violations of assumptions aside (for instance that the test set samples are not independent), as Thomas Dietterich noted based on empircal results in a simulated study [1], this test tends to have a high false positive rate (here: incorrectly detecting  difference when there is none), which is among the reasons why it is not recommended in practice.

Nonetheless, for the sake of completeness, and since it a commonly used method in practice, the general procedure is outlined below as follows (which also generally applies to the different hypothesis tests presented later): 

1. formulate the hypothesis to be tested (for instance, the null hypothesis stating that the proportions are the same; consequently, the alternative hypothesis that the proportions are different, if we use a two-tailed test);
2. decide upon a significance threshold (for instance, if the probability of observing a difference more extreme than the one observed is more than 5%, then we plan to reject the null hypothesis); 
3. analyze the data, compute the test statistic (here: z-score), and compare its associated p-value (probability) to the previously determined significance threshold;
4. based on the p-value and significance threshold, either accept or reject the null hypothesis at the given confidence level and interpret the results.



The z-score is computed as the observed difference divided by the square root for their combined variances


$$
z = \frac{ACC_1 - ACC_2}{\sqrt{\sigma_{1}^2 + \sigma_{2}^2}},
$$
where $ACC_1$ is the accuracy of one model and $ACC_2$ is the accuracy of a second model estimated from the test set. Recall that we computed the variance of the estimated of the estimated accuracy as 


$$
\sigma^2 = \frac{ACC(1-ACC)}{n}
$$


in Part I  and then computed the confidence interval (Normal Approximation Interval) as 
$$
ACC \pm z \times \sigma,
$$
where $z=1.96$ for a 95% confidence interval.  Comparing the confidence intervals of two accuracy estimates and checking whether they overlap is then analogous to computing the $z$ value for the difference in proportions and comparing the probability (p-value) to the chosen significance threshold. So, to compute the z-score directly for the difference of two proportions, $ACC_1$ and $ACC_2$, we pool these proportions (assuming that $ACC_1$ and $ACC_2$ are the performances of two models estimated on two indendent test sets of size $n_1$ and $n_2$, respectively),

$$
ACC_{1, 2} = \frac{ACC_1 \times n_1 + ACC_2 \times n_2}{n_1 + n_2},
$$

and compute the standard deviation as

$$
\sigma_{1,2} = \sqrt{\frac{ACC_{1, 2} (1 - ACC_{1, 2})}{n_1 + n_2}},
$$

such that we can compute the z-score, 


$$
z = \frac{ACC_1 - ACC_2}{\sigma_{1,2}}.
$$
Since, due to using the same test set (and violating the independence assumption) we have $n_1 = n_2 = n$, so that we can simplify the z-score computation to 

$$
z =  \frac{ACC_1 - ACC_2}{\sqrt{2\sigma^2}} = \frac{ACC_1 - ACC_2}{\sqrt{2\cdot ACC_{1,2}(1-ACC_{1,2}))/n}}.
$$

where $ACC_{1, 2}$ is simply $(ACC_1 + ACC_2)/2$.



In the second step, based on the computed $z$ value (this assumes the the test errors are independent, which is usually violated in practice as we use the same test set) we can reject the null hypothesis that the a pair of models has equal performance (here, measured in "classification aaccuracy") at an $\alpha=0.05$ level if $z$ is greater than 1.96. Or if we want to put in the extra work, we can compute the area under the a standard normal cumulative distribution at the z-score threshold. If we find this p-value is smaller than a significance level we set prior to conducting the test, then we can reject the null hypothesis at that given significance level.

The problem with this test though is that we use the same test set to compute the accuracy of the two classifiers; thus, it might be better to use a paired test such as a paired sample t-test, but a more robust alternative is the McNemar test.

### References

- [1] Dietterich, Thomas G. "Approximate statistical tests for comparing supervised classification learning algorithms." *Neural computation* 10, no. 7 (1998): 1895-1923.

## Example 1 - Difference of Proportions

As an example for applying this test, consider the following 2 model predictions:


```python
import numpy as np

## Dataset:

# ground truth labels of the test dataset:

y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0])


# predictions by 3 classifiers (`y_model_1`, `y_model_2`, and `y_model_3`):

y_model_1 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0])

y_model_2 = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0])
```

Assume, the test accuracies are as follows:


```python
acc_1 = np.sum(y_true == y_model_1) / y_true.shape[0]
acc_2 = np.sum(y_true == y_model_2) / y_true.shape[0]

print('Accuracy Model 1:', acc_1)
print('Accuracy Model 2:', acc_2)

```

    Accuracy Model 1: 0.84
    Accuracy Model 2: 0.92


Now, setting a significance threshold of $\alpha=0.05$ and conducting the test


```python
from mlxtend.evaluate import proportion_difference

z, p_value = proportion_difference(acc_1, acc_2, n_1=y_true.shape[0])

print('z: %.3f' % z)
print('p-value: %.3f' % p_value)
```

    z: -1.754
    p-value: 0.040


we find that there is a statistically significant difference between the model performances. It should be highlighted though that using this test, due to the typical independence violation of using the same test set as well as its high false positive rate, it is not recommended to use this test in practice.

## API


*proportion_difference(proportion_1, proportion_2, n_1, n_2=None)*

Computes the test statistic and p-value for a difference of
proportions test.

**Parameters**

- `proportion_1` : float

    The first proportion

- `proportion_2` : float

    The second proportion

- `n_1` : int

    The sample size of the first test sample

- `n_2` : int or None (default=None)

    The sample size of the second test sample.
    If `None`, `n_1`=`n_2`.

**Returns**


- `z, p` : float or None, float

    Returns the z-score and the p-value


**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/](http://rasbt.github.io/mlxtend/user_guide/evaluate/proportion_difference/)


