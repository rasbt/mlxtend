# Cochran's Q Test

Cochran's Q test for comparing the performance of multiple classifiers.

> `from mlxtend.evaluate import cochrans_q`    

## Overview

Cochran's Q test can be regarded as a generalized version of McNemar's test that can be applied to evaluate multiple classifiers. In a sense, Cochran's Q test is analogous to ANOVA for binary outcomes. 

To compare more than two classifiers, we can use Cochran's Q test, which has a test statistic $Q$ that is approximately, (similar to McNemar's test), distributed as chi-squared with $L-1$ degrees of freedom, where L is the number of models we evaluate (since $L=2$ for McNemar's test, McNemars test statistic approximates a chi-squared distribution with one degree of freedom). 

More formally, Cochran's Q test tests the hypothesis that there is no difference between the classification accuracies [1]: 

$$p_i: H_0 = p_1 = p_2 = \cdots = p_L.$$

Let $\{D_1, \dots , D_L\}$ be a set of classifiers who have all been tested on the same dataset. If the L classifiers don't perform differently, then the following Q statistic is distributed approximately as
"chi-squared" with $L-1$ degrees of freedom:

$$Q_C = (L-1) \frac{L \sum^{L}_{i=1}G_{i}^{2} - T^2}{LT - \sum^{N_{ts}}_{j=1} (L_j)^2}.$$

Here, $G_i$ is the number of objects out of $N_{ts}$ correctly classified by $D_i= 1, \dots  L$; $L_j$ is the number of classifiers out of $L$ that correctly classified object $\mathbf{z}_j \in \mathbf{Z}_{ts}$, where $\mathbf{Z}_{ts} = \{\mathbf{z}_1, ... \mathbf{z}_{N_{ts}}\}$ is the test dataset on which the classifers are tested on; and $T$ is the total number of correct number of votes among the $L$ classifiers [2]:

$$ T = \sum_{i=1}^{L} G_i = \sum^{N_{ts}}_{j=1} L_j.$$


To perform Cochran's Q test, we typically organize the classificier predictions in a binary $N_{ts} \times L$ matrix. The $ij\text{th}$ entry of such matrix is 0 if a classifier $D_j$ has misclassified a data example (vector) $\mathbf{z}_i$ and 1 otherwise (if the classifier predicted the class label $l(\mathbf{z}_i)$ correctly) [2].

The following example taken from [2] illustrates how the classification results may be organized. For instance, assume we have the ground truth labels of the test dataset `y_true` and the following predictions by 3 classifiers (`y_model_1`, `y_model_2`, and `y_model_3`):

    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0])

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

The table of correct (1) and incorrect (0) classifications may then look as follows:

|          | $D_1$ (model 1)   | $D_2$ (model 2)   | $D_3$ (model 3)   | Occurrences |
|----------|-------------------|-------------------|-------------------|-------------|
|          | 1                 | 1                 | 1                 | 80          |
|          | 1                 | 1                 | 0                 | 2           |
|          | 1                 | 0                 | 1                 | 0           |
|          | 1                 | 0                 | 0                 | 2           |
|          | 0                 | 1                 | 1                 | 9           |
|          | 0                 | 1                 | 0                 | 1           |
|          | 0                 | 0                 | 1                 | 3           |
|          | 0                 | 0                 | 0                 | 3           |
| Accuracy | 84/100*100% = 84% | 92/100*100% = 92% | 92/100*100% = 92% |             |

By plugging in the respective value into the previous equation, we obtain the following $Q$ value [2]:

$$Q_c = 2 \times \frac{3 \times (84^2 + 92^2 + 92^2) - 268^2}{3\times 268-(80 \times 9 + 11 \times 4 + 6 \times 1)} \approx 7.5294.$$

(Note that the $Q$ value in [2] is listed as 3.7647 due to a typo as discussed with the author, the value 7.5294 is the correct one.)

Now, the Q value (approximating $\chi^2$) corresponds to a p-value of approx. 0.023 assuming a $\chi^2$ distribution with $L-1 = 2$ degrees of freedom. Assuming that we chose a significance level of $\alpha=0.05$, we would reject the null hypothesis that all classifiers perform equally well, since $0.023 < \alpha$.

In practice, if we successfully rejected the null hypothesis, we could perform multiple post hoc pair-wise tests -- for example, McNemar tests with a Bonferroni correction -- to determine which pairs have different population proportions.


### References

- [1] Fleiss, Joseph L., Bruce Levin, and Myunghee Cho Paik. Statistical methods for rates and proportions. John Wiley & Sons, 2013.
- [2] Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.



## Example 1 - Cochran's Q test


```python
import numpy as np
from mlxtend.evaluate import cochrans_q
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar

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
q, p_value = cochrans_q(y_true, 
                        y_model_1, 
                        y_model_2, 
                        y_model_3)

print('Q: %.3f' % q)
print('p-value: %.3f' % p_value)
```

    Q: 7.529
    p-value: 0.023


Since the p-value is smaller than $\alpha$, we can reject the null hypothesis and conclude that there is a difference between the classification accuracies. As mentioned in the introduction earlier, we could now perform multiple post hoc pair-wise tests -- for example, McNemar tests with a Bonferroni correction -- to determine which pairs have different population proportions.

Lastly, let's illustrate that Cochran's Q test is indeed just a generalized version of McNemar's test:


```python
chi2, p_value = cochrans_q(y_true, 
                           y_model_1, 
                           y_model_2)

print('Cochran\'s Q Chi^2: %.3f' % chi2)
print('Cochran\'s Q p-value: %.3f' % p_value)
```

    Cochran's Q Chi^2: 5.333
    Cochran's Q p-value: 0.021



```python
chi2, p_value = mcnemar(mcnemar_table(y_true, 
                                      y_model_1, 
                                      y_model_2),
                        corrected=False)

print('McNemar\'s Chi^2: %.3f' % chi2)
print('McNemar\'s p-value: %.3f' % p_value)
```

    McNemar's Chi^2: 5.333
    McNemar's p-value: 0.021


## API


*cochrans_q(y_target, *y_model_predictions)*

Cochran's Q test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `q, p` : float or None, float

    Returns the Q (chi-squared) value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/](http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/)


