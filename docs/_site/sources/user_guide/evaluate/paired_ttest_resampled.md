# Resampled paired *t* test

Resampled paired *t* test procedure to compare the performance of two models

> `from mlxtend.evaluate import paired_ttest_resample`    

## Overview

Resampled paired *t* test procedure (also called k-hold-out paired *t* test) is a popular method for comparing the performance of two models (classifiers or regressors); however, this method has many drawbacks and is not recommended to be used in practice [1], and techniques such as the [`paired_ttest_5x2cv`](paired_ttest_5x2cv.md) should be used instead.

To explain how this method works, let's consider to estimator (e.g., classifiers) A and B. Further, we have a labeled dataset *D*. In the common hold-out method, we typically split the dataset into 2 parts: a training and a test set. In the resampled paired *t* test procedure, we repeat this splitting procedure (with typically 2/3 training data and 1/3 test data) *k* times (usually 30). In each iteration, we train A and B on the training set and evaluate it on the test set. Then, we compute the difference in performance between A and B in each iteration so that we obtain *k* difference measures. Now, by making the assumption that these *k* differences were independently drawn and follow an approximately normal distribution, we can compute the following *t* statistic with *k-1* degrees of freedom according to Student's *t* test, under the null hypothesis that the models A and B have equal performance:

$$t = \frac{\overline{p} \sqrt{k}}{\sqrt{\sum_{i=1}^{k}(p^{(i) - \overline{p}})^2 / (k-1)}}.$$

Here, $p^{(i)}$ computes the difference between the model performances in the $i$th iteration, $p^{(i)} = p^{(i)}_A - p^{(i)}_B$, and $\overline{p}$ represents the average difference between the classifier performances, $\overline{p} = \frac{1}{k} \sum^k_{i=1} p^{(i)}$.

Once we computed the *t* statistic we can compute the p value and compare it to our chosen significance level, e.g., $\alpha=0.05$. If the p value is smaller than $\alpha$, we reject the null hypothesis and accept that there is a significant difference in the two models.

To summarize the procedure:

0. i := 0
1. while i < k:
  2. split dataset into training and test subsets
  3. fit models A and B to the training set
  4. compute the performances of A and B on the test set
  5. record the performance difference between A and B
  6. i := i + 1
3. compute t-statistic
4. compute p value from t-statistic with k-1 degrees of freedom
5. compare p value to chosen significance threshold



The problem with this method, and the reason why it is not recommended to be used in practice, is that it violates the assumptions of Student's *t* test [1]:

- the difference between the model performances ($p^{(i)} = p^{(i)}_A - p^{(i)}_B$) are not normal distributed because $p^{(i)}_A$ and $p^{(i)}_B$ are not independent
- the $p^{(i)}$'s themselves are not independent because of the overlapping test sets; also, test and training sets overlap as well

### References

- [1] Dietterich TG (1998) Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. *Neural Comput* 10:1895–1923.

## Example 1 - Resampled paired *t* test

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

Now, let's assume a significance threshold of $\alpha=0.05$ for rejecting the null hypothesis that both algorithms perform equally well on the dataset and conduct the paired sample *t* test:


```python
from mlxtend.evaluate import paired_ttest_resampled


t, p = paired_ttest_resampled(estimator1=clf1,
                              estimator2=clf2,
                              X=X, y=y,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
```

    t statistic: -1.809
    p value: 0.081


Since $p > t$, we cannot reject the null hypothesis and may conclude that the performance of the two algorithms is not significantly different. 

While it is generally not recommended to apply statistical tests multiple times without correction for multiple hypothesis testing, let us take a look at an example where the decision tree algorithm is limited to producing a very simple decision boundary that would result in a relatively bad performance:


```python
clf2 = DecisionTreeClassifier(random_state=1, max_depth=1)

score2 = clf2.fit(X_train, y_train).score(X_test, y_test)
print('Decision tree accuracy: %.2f%%' % (score2*100))


t, p = paired_ttest_resampled(estimator1=clf1,
                              estimator2=clf2,
                              X=X, y=y,
                              random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
```

    Decision tree accuracy: 63.16%
    t statistic: 39.214
    p value: 0.000


Assuming that we conducted this test also with a significance level of $\alpha=0.05$, we can reject the null-hypothesis that both models perform equally well on this dataset, since the p-value ($p < 0.001$) is smaller than $\alpha$.

## API


*paired_ttest_resampled(estimator1, estimator2, X, y, num_rounds=30, test_size=0.3, scoring=None, random_seed=None)*

Implements the resampled paired t test procedure
to compare the performance of two models
(also called k-hold-out paired t test).

**Parameters**

- `estimator1` : scikit-learn classifier or regressor



- `estimator2` : scikit-learn classifier or regressor



- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.


- `y` : array-like, shape = [n_samples]

    Target values.


- `num_rounds` : int (default: 30)

    Number of resampling iterations
    (i.e., train/test splits)


- `test_size` : float or int (default: 0.3)

    If float, should be between 0.0 and 1.0 and
    represent the proportion of the dataset to use
    as a test set.
    If int, represents the absolute number of test exsamples.


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
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/](http://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest_resampled/)


