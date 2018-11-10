# 5x2cv paired *t* test

5x2cv paired *t* test procedure to compare the performance of two models

> `from mlxtend.evaluate import paired_ttest_5x2cv`    

## Overview

The 5x2cv paired *t* test is a procedure for comparing the performance of two models (classifiers or regressors)
that was proposed by Dietterich [1] to address shortcomings in other methods such as the resampled paired *t* test (see [`paired_ttest_resampled`](paired_ttest_resampled.md)) and the k-fold cross-validated paired *t* test (see [`paired_ttest_kfold_cv`](paired_ttest_kfold_cv.md)).

To explain how this method works, let's consider to estimator (e.g., classifiers) A and B. Further, we have a labeled dataset *D*. In the common hold-out method, we typically split the dataset into 2 parts: a training and a test set. In the 5x2cv paired *t* test, we repeat the splitting (50% training and 50% test data) 5 times. 

In each of the 5 iterations, we fit A and B to the training split and evaluate their performance ($p_A$ and $p_B$) on the test split. Then, we rotate the training and test sets (the training set becomes the test set and vice versa) compute the performance again, which results in 2 performance difference measures:

$$p^{(1)} = p^{(1)}_A - p^{(1)}_B$$

and

$$p^{(2)} = p^{(2)}_A - p^{(2)}_B.$$

Then, we estimate the estimate mean and variance of the differences:

$\overline{p} = \frac{p^{(1)} + p^{(2)}}{2}$

and

$s^2 = (p^{(1)} - \overline{p})^2 + (p^{(2)} - \overline{p})^2.$

The variance of the difference is computed for the 5 iterations and then used to compute the *t* statistic as follows:

$$t = \frac{p_1^{(1)}}{\sqrt{(1/5) \sum_{i=1}^{5}s_i^2}},$$

where $p_1^{(1)}$ is the $p_1$ from the very first iteration. The *t* statistic, assuming that it approximately follows as *t* distribution with 5 degrees of freedom, under the null hypothesis that the models A and B have equal performance. Using the *t* statistic, the p value can be computed and compared with a previously chosen significance level, e.g., $\alpha=0.05$. If the p value is smaller than $\alpha$, we reject the null hypothesis and accept that there is a significant difference in the two models.



### References

- [1] Dietterich TG (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Comput* 10:1895–1923.

## Example 1 - 5x2cv paired *t* test

Assume we want to compare two classification algorithms, logistic regression and a decision tree algorithm:


```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import train_test_split


X, y = iris_data()
clf1 = LogisticRegression(random_state=1)
clf2 = DecisionTreeClassifier(random_state=1)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.25,
                     random_state=123)

score1 = clf1.fit(X_train, y_train).score(X_test, y_test)
score2 = clf2.fit(X_train, y_train).score(X_test, y_test)

print('Logistic regression accuracy: %.2f%%' % (score1*100))
print('Decision tree accuracy: %.2f%%' % (score2*100))
```

    Logistic regression accuracy: 97.37%
    Decision tree accuracy: 94.74%


Note that these accuracy values are not used in the paired *t* test procedure as new test/train splits are generated during the resampling procedure, the values above are just serving the purpose of intuition.

Now, let's assume a significance threshold of $\alpha=0.05$ for rejecting the null hypothesis that both algorithms perform equally well on the dataset and conduct the 5x2cv *t* test:


```python
from mlxtend.evaluate import paired_ttest_5x2cv


t, p = paired_ttest_5x2cv(estimator1=clf1,
                          estimator2=clf2,
                          X=X, y=y,
                          random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
```

    t statistic: -1.539
    p value: 0.184


Since $p > \alpha$, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different. 

While it is generally not recommended to apply statistical tests multiple times without correction for multiple hypothesis testing, let us take a look at an example where the decision tree algorithm is limited to producing a very simple decision boundary that would result in a relatively bad performance:


```python
clf2 = DecisionTreeClassifier(random_state=1, max_depth=1)

score2 = clf2.fit(X_train, y_train).score(X_test, y_test)
print('Decision tree accuracy: %.2f%%' % (score2*100))


t, p = paired_ttest_5x2cv(estimator1=clf1,
                          estimator2=clf2,
                          X=X, y=y,
                          random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
```

    Decision tree accuracy: 63.16%
    t statistic: 5.386
    p value: 0.003


Assuming that we conducted this test also with a significance level of $\alpha=0.05$, we can reject the null-hypothesis that both models perform equally well on this dataset, since the p-value ($p < 0.001$) is smaller than $\alpha$.

## API


*paired_ttest_5x2cv(estimator1, estimator2, X, y, scoring=None, random_seed=None)*

Implements the 5x2cv paired t test proposed
by Dieterrich (1998)
to compare the performance of two models.

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


- `scoring` : str, callable, or None (default: None)

    If None (default), uses 'accuracy' for sklearn classifiers
    and 'r2' for sklearn regressors.
    If str, uses a sklearn scoring metric string identifier, for example
    {accuracy, f1, precision, recall, roc_auc} for classifiers,
    {'mean_absolute_error', 'mean_squared_error'/'neg_mean_squared_error',
    'median_absolute_error', 'r2'} for regressors.
    If a callable object or function is provided, it has to be conform with
    sklearn's signature ``scorer(estimator, X, y)``; see
    http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    for more information.


- `random_seed` : int or None (default: None)

    Random seed for creating the test/train splits.

**Returns**

- `t` : float

    The t-statistic


- `pvalue` : float

    Two-tailed p-value.
    If the chosen significance level is larger
    than the p-value, we reject the null hypothesis
    and accept that there are significant differences
    in the two compared models.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_5x2cv/)


