# F-Test

F-test for comparing the performance of multiple classifiers.

> `from mlxtend.evaluate import ftest`    

## Overview

In the context of evaluating machine learning models, the F-test by George W. Snedecor [1] can be regarded as analogous to Cochran's Q test that can be applied to evaluate multiple classifiers (i.e., whether their accuracies estimated on a test set differ) as described by Looney [2][3]. 

More formally, assume the task to test the null hypothesis that there is no difference between the classification accuracies [1]: 

$$p_i: H_0 = p_1 = p_2 = \cdots = p_L.$$

Let $\{D_1, \dots , D_L\}$ be a set of classifiers who have all been tested on the same dataset. If the L classifiers don't perform differently, then the F statistic is distributed according to an F distribution with $(L-1$) and $(L-1)\times N$ degrees of freedom, where $N$ is the number of examples in the test set.

The calculation of the F statistic consists of several components, which are listed below (adopted from [3]).

Sum of squares of the classifiers:

$$
SSA = N \sum_{i=1}^{N} (L_j)^2,
$$


where $L_j$ is the number of classifiers out of $L$ that correctly classified object $\mathbf{z}_j \in \mathbf{Z}_{N}$, where $\mathbf{Z}_{N} = \{\mathbf{z}_1, ... \mathbf{z}_{N}\}$ is the test dataset on which the classifers are tested on.

The sum of squares for the objects:

$$
SSB= \frac{1}{L} \sum_{j=1}^N (L_j)^2 - L\cdot N \cdot ACC_{avg}^2,
$$

where $ACC_{avg}$ is the average of the accuracies of the different models $ACC_{avg} = \sum_{i=1}^L ACC_i$.

The total sum of squares:

$$
SST = L\cdot N \cdot ACC_{avg}^2 (1 - ACC_{avg}^2).
$$

The sum of squares for the classification--object interaction:

$$
SSAB = SST - SSA - SSB.
$$

The mean SSA and mean SSAB values:

$$
MSA = \frac{SSA}{L-1},
$$

and

$$
MSAB = \frac{SSAB}{(L-1) (N-1)}.
$$

From the MSA and MSAB, we can then calculate the F-value as

$$
F = \frac{MSA}{MSAB}.
$$


After computing the F-value, we can then look up the p-value from a F-distribution table for the corresponding degrees of freedom or obtain it computationally from a cumulative F-distribution function. In practice, if we successfully rejected the null hypothesis at a previously chosen significance threshold, we could perform multiple post hoc pair-wise tests -- for example, McNemar tests with a Bonferroni correction -- to determine which pairs have different population proportions.


### References

- [1]  Snedecor, George W. and Cochran, William G. (1989), Statistical Methods, Eighth Edition, Iowa State University Press.
- [2] Looney, Stephen W. "A statistical technique for comparing the accuracies of several classifiers." Pattern Recognition Letters 8, no. 1 (1988): 5-9.
- [3] Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.



## Example 1 - F-test


```python
import numpy as np
from mlxtend.evaluate import ftest

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

y_model_3 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                      1, 1])
```

Assuming a significance level $\alpha=0.05$, we can conduct Cochran's Q test as follows, to test the null hypothesis there is no difference between the classification accuracies, $p_i: H_0 = p_1 = p_2 = \cdots = p_L$:


```python
f, p_value = ftest(y_true, 
                   y_model_1, 
                   y_model_2, 
                   y_model_3)

print('F: %.3f' % f)
print('p-value: %.3f' % p_value)
```

    F: 3.873
    p-value: 0.022


Since the p-value is smaller than $\alpha$, we can reject the null hypothesis and conclude that there is a difference between the classification accuracies. As mentioned in the introduction earlier, we could now perform multiple post hoc pair-wise tests -- for example, McNemar tests with a Bonferroni correction -- to determine which pairs have different population proportions.

## API


*ftest(y_target, *y_model_predictions)*

F-Test test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `f, p` : float or None, float

    Returns the F-value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/](http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/)


