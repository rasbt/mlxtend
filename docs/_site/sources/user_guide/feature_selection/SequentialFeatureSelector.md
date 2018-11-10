# Sequential Feature Selector

Implementation of *sequential feature algorithms* (SFAs) -- greedy search algorithms -- that have been developed as a suboptimal solution to the computationally often not feasible exhaustive search.

> from mlxtend.feature_selection import SequentialFeatureSelector

## Overview

Sequential feature selection algorithms are a family of greedy search algorithms that are used to reduce an initial *d*-dimensional feature space to a *k*-dimensional feature subspace where *k < d*. The motivation behind feature selection algorithms is to automatically select a subset of features that is most relevant to the problem. The goal of feature selection is two-fold: We want to improve the computational efficiency and reduce the generalization error of the model by removing irrelevant features or noise. A wrapper approach such as sequential feature selection is especially useful if embedded feature selection -- for example, a regularization penalty like LASSO -- is not applicable.

In a nutshell, SFAs remove or add one feature at the time based on the classifier performance until a feature subset of the desired size *k* is reached. There are 4 different flavors of SFAs available via the `SequentialFeatureSelector`:

1. Sequential Forward Selection (SFS)
2. Sequential Backward Selection (SBS)
3. Sequential Forward Floating Selection (SFFS)
4. Sequential Backward Floating Selection (SBFS)

The ***floating*** variants, SFFS and SBFS, can be considered as extensions to the simpler SFS and SBS algorithms. The floating algorithms have an additional exclusion or inclusion step to remove features once they were included (or excluded), so that a larger number of feature subset combinations can be sampled. It is important to emphasize that this step is conditional and only occurs if the resulting feature subset is assessed as "better" by the criterion function after removal (or addition) of a particular feature. Furthermore, I added an optional check to skip the conditional exclusion steps if the algorithm gets stuck in cycles.  


---

How is this different from *Recursive Feature Elimination* (RFE)  -- e.g., as implemented in `sklearn.feature_selection.RFE`? RFE is computationally less complex using the feature weight coefficients (e.g., linear models) or feature importance (tree-based algorithms) to eliminate features recursively, whereas SFSs eliminate (or add) features based on a user-defined classifier/regression performance metric.

---

The SFAs  are outlined in pseudo code below:

### Sequential Forward Selection (SFS)


**Input:** $Y = \{y_1, y_2, ..., y_d\}$  

- The ***SFS*** algorithm takes the whole $d$-dimensional feature set as input.


**Output:** $X_k = \{x_j \; | \;j = 1, 2, ..., k; \; x_j \in Y\}$, where $k = (0, 1, 2, ..., d)$

- SFS returns a subset of features; the number of selected features $k$, where $k < d$, has to be specified *a priori*.

**Initialization:** $X_0 = \emptyset$, $k = 0$

- We initialize the algorithm with an empty set $\emptyset$ ("null set") so that $k = 0$ (where $k$ is the size of the subset).

**Step 1 (Inclusion):**  

  $x^+ = \text{ arg max } J(x_k + x), \text{ where }  x \in Y - X_k$  
  $X_{k+1} = X_k + x^+$  
  $k = k + 1$    
  *Go to Step 1* 

- in this step, we add an additional feature, $x^+$, to our feature subset $X_k$.
- $x^+$ is the feature that maximizes our criterion function, that is, the feature that is associated with the best classifier performance if it is added to $X_k$.
- We repeat this procedure until the termination criterion is satisfied.

**Termination:** $k = p$

- We add features from the feature subset $X_k$ until the feature subset of size $k$ contains the number of desired features $p$ that we specified *a priori*.

### Sequential Backward Selection (SBS)

**Input:** the set of all features, $Y = \{y_1, y_2, ..., y_d\}$  

- The SBS algorithm takes the whole feature set as input.

**Output:** $X_k = \{x_j \; | \;j = 1, 2, ..., k; \; x_j \in Y\}$, where $k = (0, 1, 2, ..., d)$

- SBS returns a subset of features; the number of selected features $k$, where $k < d$, has to be specified *a priori*.

**Initialization:** $X_0 = Y$, $k = d$

- We initialize the algorithm with the given feature set so that the $k = d$.


**Step 1 (Exclusion):**  

$x^- = \text{ arg max } J(x_k - x), \text{  where } x \in X_k$  
$X_{k-1} = X_k - x^-$  
$k = k - 1$  
*Go to Step 1*  

- In this step, we remove a feature, $x^-$ from our feature subset $X_k$.
- $x^-$ is the feature that maximizes our criterion function upon re,oval, that is, the feature that is associated with the best classifier performance if it is removed from $X_k$.
- We repeat this procedure until the termination criterion is satisfied.


**Termination:** $k = p$

- We add features from the feature subset $X_k$ until the feature subset of size $k$ contains the number of desired features $p$ that we specified *a priori*.



### Sequential Backward Floating Selection (SBFS)

**Input:** the set of all features, $Y = \{y_1, y_2, ..., y_d\}$  

- The SBFS algorithm takes the whole feature set as input.

**Output:** $X_k = \{x_j \; | \;j = 1, 2, ..., k; \; x_j \in Y\}$, where $k = (0, 1, 2, ..., d)$

- SBFS returns a subset of features; the number of selected features $k$, where $k < d$, has to be specified *a priori*.

**Initialization:** $X_0 = Y$, $k = d$

- We initialize the algorithm with the given feature set so that the $k = d$.

**Step 1 (Exclusion):**  

$x^- = \text{ arg max } J(x_k - x), \text{  where } x \in X_k$  
$X_{k-1} = X_k - x^-$  
$k = k - 1$  
*Go to Step 2*  

- In this step, we remove a feature, $x^-$ from our feature subset $X_k$.
- $x^-$ is the feature that maximizes our criterion function upon re,oval, that is, the feature that is associated with the best classifier performance if it is removed from $X_k$.


**Step 2 (Conditional Inclusion):**  
<br>
$x^+ = \text{ arg max } J(x_k + x), \text{ where } x \in Y - X_k$  
*if J(x_k + x) > J(x_k + x)*:    
&nbsp;&nbsp;&nbsp;&nbsp; $X_{k+1} = X_k + x^+$  
&nbsp;&nbsp;&nbsp;&nbsp; $k = k + 1$  
*Go to Step 1*  

- In Step 2, we search for features that improve the classifier performance if they are added back to the feature subset. If such features exist, we add the feature $x^+$ for which the performance improvement is maximized. If $k = 2$ or an improvement cannot be made (i.e., such feature $x^+$ cannot be found), go back to step 1; else, repeat this step.


**Termination:** $k = p$

- We add features from the feature subset $X_k$ until the feature subset of size $k$ contains the number of desired features $p$ that we specified *a priori*.


### Sequential Forward Floating Selection (SFFS)

**Input:** the set of all features, $Y = \{y_1, y_2, ..., y_d\}$  

- The ***SFFS*** algorithm takes the whole feature set as input, if our feature space consists of, e.g. 10, if our feature space consists of 10 dimensions (***d = 10***).
<br><br>

**Output:** a subset of features, $X_k = \{x_j \; | \;j = 1, 2, ..., k; \; x_j \in Y\}$, where $k = (0, 1, 2, ..., d)$

- The returned output of the algorithm is a subset of the feature space of a specified size. E.g., a subset of 5 features from a 10-dimensional feature space (***k = 5, d = 10***).
<br><br>

**Initialization:** $X_0 = Y$, $k = d$

- We initialize the algorithm with an empty set ("null set") so that the ***k = 0*** (where ***k*** is the size of the subset)
<br><br>

**Step 1 (Inclusion):**  
<br>
&nbsp;&nbsp;&nbsp;&nbsp; $x^+ = \text{ arg max } J(x_k + x), \text{ where }  x \in Y - X_k$  
&nbsp;&nbsp;&nbsp;&nbsp; $X_{k+1} = X_k + x^+$  
&nbsp;&nbsp;&nbsp;&nbsp; $k = k + 1$    
&nbsp;&nbsp;&nbsp;&nbsp;*Go to Step 2*  
<br> <br>
**Step 2 (Conditional Exclusion):**  
<br>
&nbsp;&nbsp;&nbsp;&nbsp; $x^- = \text{ arg max } J(x_k - x), \text{ where } x \in X_k$  
&nbsp;&nbsp;&nbsp;&nbsp;$if \; J(x_k - x) > J(x_k - x)$:    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $X_{k-1} = X_k - x^- $  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; $k = k - 1$    
&nbsp;&nbsp;&nbsp;&nbsp;*Go to Step 1*  

- In step 1, we include the feature from the ***feature space*** that leads to the best performance increase for our ***feature subset*** (assessed by the ***criterion function***). Then, we go over to step 2
- In step 2, we only remove a feature if the resulting subset would gain an increase in performance. If $k = 2$ or an improvement cannot be made (i.e., such feature $x^+$ cannot be found), go back to step 1; else, repeat this step.


- Steps 1 and 2 are repeated until the **Termination** criterion is reached.
<br><br>

**Termination:** stop when ***k*** equals the number of desired features


### References

- Ferri, F. J., Pudil P., Hatef, M., Kittler, J. (1994). [*"Comparative study of techniques for large-scale feature selection."*](https://books.google.com/books?hl=en&lr=&id=sbajBQAAQBAJ&oi=fnd&pg=PA403&dq=comparative+study+of+techniques+for+large+scale&ots=KdIOYpA8wj&sig=hdOsBP1HX4hcDjx4RLg_chheojc#v=onepage&q=comparative%20study%20of%20techniques%20for%20large%20scale&f=false) Pattern Recognition in Practice IV : 403-413.

- Pudil, P., Novovičová, J., & Kittler, J. (1994). [*"Floating search methods in feature selection."*](http://www.sciencedirect.com/science/article/pii/0167865594901279) Pattern recognition letters 15.11 (1994): 1119-1125.

## Example 1 - A simple Sequential Forward Selection example

Initializing a simple classifier from scikit-learn:


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=4)
```

We start by selection the "best" 3 features from the Iris dataset via Sequential Forward Selection (SFS). Here, we set `forward=True` and `floating=False`. By choosing `cv=0`, we don't perform any cross-validation, therefore, the performance (here: `'accuracy'`) is computed entirely on the training set. 


```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=0)

sfs1 = sfs1.fit(X, y)
```

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 1/3 -- score: 0.96[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 2/3 -- score: 0.973333333333[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 3/3 -- score: 0.973333333333

Via the `subsets_` attribute, we can take a look at the selected feature indices at each step:


```python
sfs1.subsets_
```




    {1: {'avg_score': 0.95999999999999996,
      'cv_scores': array([ 0.96]),
      'feature_idx': (3,),
      'feature_names': ('3',)},
     2: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (2, 3),
      'feature_names': ('2', '3')},
     3: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (1, 2, 3),
      'feature_names': ('1', '2', '3')}}



Note that the 'feature_names' entry is simply a string representation of the 'feature_idx' in this case. Optionally, we can provide custom feature names via the `fit` method's `custom_feature_names` parameter:


```python
feature_names = ('sepal length', 'sepal width', 'petal length', 'petal width')
sfs1 = sfs1.fit(X, y, custom_feature_names=feature_names)
sfs1.subsets_
```

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 1/3 -- score: 0.96[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 2/3 -- score: 0.973333333333[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:16] Features: 3/3 -- score: 0.973333333333




    {1: {'avg_score': 0.95999999999999996,
      'cv_scores': array([ 0.96]),
      'feature_idx': (3,),
      'feature_names': ('petal width',)},
     2: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (2, 3),
      'feature_names': ('petal length', 'petal width')},
     3: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (1, 2, 3),
      'feature_names': ('sepal width', 'petal length', 'petal width')}}



Furthermore, we can access the indices of the 3 best features directly via the `k_feature_idx_` attribute:


```python
sfs1.k_feature_idx_
```




    (1, 2, 3)



And similarly, to obtain the names of these features, given that we provided an argument to the `custom_feature_names` parameter, we can refer to the `sfs1.k_feature_names_` attribute:


```python
sfs1.k_feature_names_
```




    ('sepal width', 'petal length', 'petal width')



Finally, the prediction score for these 3 features can be accesses via `k_score_`:


```python
sfs1.k_score_
```




    0.97333333333333338



## Example 2 - Toggling between SFS, SBS, SFFS, and SBFS

Using the `forward` and `floating` parameters, we can toggle between SFS, SBS, SFFS, and SBFS as shown below. Note that we are performing (stratified) 4-fold cross-validation for more robust estimates in contrast to Example 1. Via `n_jobs=-1`, we choose to run the cross-validation on all our available CPU cores.


```python
# Sequential Forward Selection
sfs = SFS(knn, 
          k_features=3, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=4,
          n_jobs=-1)
sfs = sfs.fit(X, y)

print('\nSequential Forward Selection (k=3):')
print(sfs.k_feature_idx_)
print('CV Score:')
print(sfs.k_score_)

###################################################

# Sequential Backward Selection
sbs = SFS(knn, 
          k_features=3, 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          cv=4,
          n_jobs=-1)
sbs = sbs.fit(X, y)

print('\nSequential Backward Selection (k=3):')
print(sbs.k_feature_idx_)
print('CV Score:')
print(sbs.k_score_)

###################################################

# Sequential Forward Floating Selection
sffs = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=True, 
           scoring='accuracy',
           cv=4,
           n_jobs=-1)
sffs = sffs.fit(X, y)

print('\nSequential Forward Floating Selection (k=3):')
print(sffs.k_feature_idx_)
print('CV Score:')
print(sffs.k_score_)

###################################################

# Sequential Backward Floating Selection
sbfs = SFS(knn, 
           k_features=3, 
           forward=False, 
           floating=True, 
           scoring='accuracy',
           cv=4,
           n_jobs=-1)
sbfs = sbfs.fit(X, y)

print('\nSequential Backward Floating Selection (k=3):')
print(sbfs.k_feature_idx_)
print('CV Score:')
print(sbfs.k_score_)
```

    
    Sequential Forward Selection (k=3):
    (1, 2, 3)
    CV Score:
    0.972756410256
    
    Sequential Backward Selection (k=3):
    (1, 2, 3)
    CV Score:
    0.972756410256
    
    Sequential Forward Floating Selection (k=3):
    (1, 2, 3)
    CV Score:
    0.972756410256
    
    Sequential Backward Floating Selection (k=3):
    (1, 2, 3)
    CV Score:
    0.972756410256


In this simple scenario, selecting the best 3 features out of the 4 available features in the Iris set, we end up with similar results regardless of which sequential selection algorithms we used.

## Example 3 - Visualizing the results in DataFrames

 For our convenience, we can visualize the output from the feature selection in a pandas DataFrame format using the `get_metric_dict` method of the SequentialFeatureSelector object. The columns `std_dev` and `std_err` represent the standard deviation and standard errors of the cross-validation scores, respectively.

Below, we see the DataFrame of the Sequential Forward Selector from Example 2:


```python
import pandas as pd
pd.DataFrame.from_dict(sfs.get_metric_dict()).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_score</th>
      <th>ci_bound</th>
      <th>cv_scores</th>
      <th>feature_idx</th>
      <th>feature_names</th>
      <th>std_dev</th>
      <th>std_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.952991</td>
      <td>0.0660624</td>
      <td>[0.974358974359, 0.948717948718, 0.88888888888...</td>
      <td>(3,)</td>
      <td>(3,)</td>
      <td>0.0412122</td>
      <td>0.0237939</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.959936</td>
      <td>0.0494801</td>
      <td>[0.974358974359, 0.948717948718, 0.91666666666...</td>
      <td>(2, 3)</td>
      <td>(2, 3)</td>
      <td>0.0308676</td>
      <td>0.0178214</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.972756</td>
      <td>0.0315204</td>
      <td>[0.974358974359, 1.0, 0.944444444444, 0.972222...</td>
      <td>(1, 2, 3)</td>
      <td>(1, 2, 3)</td>
      <td>0.0196636</td>
      <td>0.0113528</td>
    </tr>
  </tbody>
</table>
</div>



Now, let's compare it to the Sequential Backward Selector:


```python
pd.DataFrame.from_dict(sbs.get_metric_dict()).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_score</th>
      <th>ci_bound</th>
      <th>cv_scores</th>
      <th>feature_idx</th>
      <th>feature_names</th>
      <th>std_dev</th>
      <th>std_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.972756</td>
      <td>0.0315204</td>
      <td>[0.974358974359, 1.0, 0.944444444444, 0.972222...</td>
      <td>(1, 2, 3)</td>
      <td>(1, 2, 3)</td>
      <td>0.0196636</td>
      <td>0.0113528</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.952991</td>
      <td>0.0372857</td>
      <td>[0.974358974359, 0.948717948718, 0.91666666666...</td>
      <td>(0, 1, 2, 3)</td>
      <td>(0, 1, 2, 3)</td>
      <td>0.0232602</td>
      <td>0.0134293</td>
    </tr>
  </tbody>
</table>
</div>



We can see that both SFS and SBFS found the same "best" 3 features, however, the intermediate steps where obviously different.

The `ci_bound` column in the DataFrames above represents the confidence interval around the computed cross-validation scores. By default, a confidence interval of 95% is used, but we can use different confidence bounds via the `confidence_interval` parameter. E.g., the confidence bounds for a 90% confidence interval can be obtained as follows:


```python
pd.DataFrame.from_dict(sbs.get_metric_dict(confidence_interval=0.90)).T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>avg_score</th>
      <th>ci_bound</th>
      <th>cv_scores</th>
      <th>feature_idx</th>
      <th>feature_names</th>
      <th>std_dev</th>
      <th>std_err</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>0.972756</td>
      <td>0.0242024</td>
      <td>[0.974358974359, 1.0, 0.944444444444, 0.972222...</td>
      <td>(1, 2, 3)</td>
      <td>(1, 2, 3)</td>
      <td>0.0196636</td>
      <td>0.0113528</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.952991</td>
      <td>0.0286292</td>
      <td>[0.974358974359, 0.948717948718, 0.91666666666...</td>
      <td>(0, 1, 2, 3)</td>
      <td>(0, 1, 2, 3)</td>
      <td>0.0232602</td>
      <td>0.0134293</td>
    </tr>
  </tbody>
</table>
</div>



## Example 4 - Plotting the results

After importing the little helper function [`plotting.plot_sequential_feature_selection`](../plotting/plot_sequential_feature_selection.md), we can also visualize the results using matplotlib figures.


```python
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt

sfs = SFS(knn, 
          k_features=4, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          verbose=2,
          cv=5)

sfs = sfs.fit(X, y)

fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()
```

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:18] Features: 1/4 -- score: 0.96[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:18] Features: 2/4 -- score: 0.966666666667[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:18] Features: 3/4 -- score: 0.953333333333[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s finished
    
    [2018-05-06 12:49:18] Features: 4/4 -- score: 0.973333333333


![png](SequentialFeatureSelector_files/SequentialFeatureSelector_43_1.png)


## Example 5 - Sequential Feature Selection for Regression

Similar to the classification examples above, the `SequentialFeatureSelector` also supports scikit-learn's estimators
for regression.


```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston

boston = load_boston()
X, y = boston.data, boston.target

lr = LinearRegression()

sfs = SFS(lr, 
          k_features=13, 
          forward=True, 
          floating=False, 
          scoring='neg_mean_squared_error',
          cv=10)

sfs = sfs.fit(X, y)
fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
```


![png](SequentialFeatureSelector_files/SequentialFeatureSelector_46_0.png)


## Example 6 -- Feature Selection with Fixed Train/Validation Splits

If you do not wish to use cross-validation (here: k-fold cross-validation, i.e., rotating training and validation folds), you can use the `PredefinedHoldoutSplit` class to specify your own, fixed training and validation split.


```python
from sklearn.datasets import load_iris
from mlxtend.evaluate import PredefinedHoldoutSplit
import numpy as np


iris = load_iris()
X = iris.data
y = iris.target

rng = np.random.RandomState(123)
my_validation_indices = rng.permutation(np.arange(150))[:30]
print(my_validation_indices)
```

    [ 72 112 132  88  37 138  87  42   8  90 141  33  59 116 135 104  36  13
      63  45  28 133  24 127  46  20  31 121 117   4]



```python
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



knn = KNeighborsClassifier(n_neighbors=4)
piter = PredefinedHoldoutSplit(my_validation_indices)

sfs1 = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=False, 
           verbose=2,
           scoring='accuracy',
           cv=piter)

sfs1 = sfs1.fit(X, y)
```

    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   4 out of   4 | elapsed:    0.0s finished
    
    [2018-09-24 02:31:21] Features: 1/3 -- score: 0.9666666666666667[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   3 out of   3 | elapsed:    0.0s finished
    
    [2018-09-24 02:31:21] Features: 2/3 -- score: 0.9666666666666667[Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=1)]: Done   2 out of   2 | elapsed:    0.0s finished
    
    [2018-09-24 02:31:21] Features: 3/3 -- score: 0.9666666666666667

## Example 7 -- Using the Selected Feature Subset For Making New Predictions


```python
# Initialize the dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=1)

knn = KNeighborsClassifier(n_neighbors=4)
```


```python
# Select the "best" three features via
# 5-fold cross-validation on the training set.

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)
sfs1 = sfs1.fit(X_train, y_train)
```


```python
print('Selected features:', sfs1.k_feature_idx_)
```

    Selected features: (1, 2, 3)



```python
# Generate the new subsets based on the selected features
# Note that the transform call is equivalent to
# X_train[:, sfs1.k_feature_idx_]

X_train_sfs = sfs1.transform(X_train)
X_test_sfs = sfs1.transform(X_test)

# Fit the estimator using the new feature subset
# and make a prediction on the test data
knn.fit(X_train_sfs, y_train)
y_pred = knn.predict(X_test_sfs)

# Compute the accuracy of the prediction
acc = float((y_test == y_pred).sum()) / y_pred.shape[0]
print('Test set accuracy: %.2f %%' % (acc * 100))
```

    Test set accuracy: 96.00 %


## Example 8 -- Sequential Feature Selection and GridSearch


```python
# Initialize the dataset

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=1)
```

Use scikit-learn's `GridSearch` to tune the hyperparameters inside and outside the `SequentialFeatureSelector`:


```python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import mlxtend

knn = KNeighborsClassifier(n_neighbors=2)

sfs1 = SFS(estimator=knn, 
           k_features=3,
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)

pipe = Pipeline([('sfs', sfs1), 
                 ('knn', knn)])

param_grid = [
  {'sfs__k_features': [1, 2, 3, 4],
   'sfs__estimator__n_neighbors': [1, 2, 3, 4]}
  ]
    
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=5,  
                  refit=False)

# run gridearch
gs = gs.fit(X_train, y_train)
```

... and the "best" parameters determined by GridSearch are ...


```python
print("Best parameters via GridSearch", gs.best_params_)
```

    Best parameters via GridSearch {'sfs__estimator__n_neighbors': 1, 'sfs__k_features': 3}


#### Obtaining the best *k* feature indices after GridSearch

If we are interested in the best *k* feature indices via `SequentialFeatureSelection.k_feature_idx_`, we have to initialize a `GridSearchCV` object with `refit=True`. Now, the grid search object will take the complete training dataset and the best parameters, which it found via cross-validation, to train the estimator pipeline.


```python
gs = GridSearchCV(estimator=pipe, 
                  param_grid=param_grid, 
                  scoring='accuracy', 
                  n_jobs=1, 
                  cv=5, 
                  refit=True)
gs = gs.fit(X_train, y_train)
```

After running the grid search, we can access the individual pipeline objects of the `best_estimator_` via the `steps` attribute.


```python
gs.best_estimator_.steps
```




    [('sfs', SequentialFeatureSelector(clone_estimator=True, cv=5,
                   estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                 metric_params=None, n_jobs=1, n_neighbors=1, p=2,
                 weights='uniform'),
                   floating=False, forward=True, k_features=3, n_jobs=1,
                   pre_dispatch='2*n_jobs', scoring='accuracy', verbose=0)),
     ('knn',
      KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                 metric_params=None, n_jobs=1, n_neighbors=2, p=2,
                 weights='uniform'))]



Via sub-indexing, we can then obtain the best-selected feature subset:


```python
print('Best features:', gs.best_estimator_.steps[0][1].k_feature_idx_)
```

    Best features: (0, 1, 3)


During cross-validation, this feature combination had a CV accuracy of:


```python
print('Best score:', gs.best_score_)
```

    Best score: 0.94



```python
gs.best_params_
```




    {'sfs__estimator__n_neighbors': 1, 'sfs__k_features': 3}



**Alternatively**, if we can set the "best grid search parameters" in our pipeline manually if we ran `GridSearchCV` with `refit=False`. It should yield the same results:


```python
pipe.set_params(**gs.best_params_).fit(X_train, y_train)
print('Best features:', pipe.steps[0][1].k_feature_idx_)
```

    Best features: (0, 1, 3)


## Example 9 -- Selecting the "best"  feature combination in a k-range

If `k_features` is set to to a tuple `(min_k, max_k)` (new in 0.4.2), the SFS will now select the best feature combination that it discovered by iterating from `k=1` to `max_k` (forward), or `max_k` to `min_k` (backward). The size of the returned feature subset is then within `max_k` to `min_k`, depending on which combination scored best during cross validation.




```python
X.shape
```




    (150, 4)




```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import wine_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X, y = wine_data()
X_train, X_test, y_train, y_test= train_test_split(X, y, 
                                                   stratify=y,
                                                   test_size=0.3,
                                                   random_state=1)

knn = KNeighborsClassifier(n_neighbors=2)

sfs1 = SFS(estimator=knn, 
           k_features=(3, 10),
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=5)

pipe = make_pipeline(StandardScaler(), sfs1)

pipe.fit(X_train, y_train)

print('best combination (ACC: %.3f): %s\n' % (sfs1.k_score_, sfs1.k_feature_idx_))
print('all subsets:\n', sfs1.subsets_)
plot_sfs(sfs1.get_metric_dict(), kind='std_err');
```

    best combination (ACC: 0.992): (0, 1, 2, 3, 6, 8, 9, 10, 11, 12)
    
    all subsets:
     {1: {'feature_idx': (6,), 'cv_scores': array([ 0.84615385,  0.6       ,  0.88      ,  0.79166667,  0.875     ]), 'avg_score': 0.7985641025641026, 'feature_names': ('6',)}, 2: {'feature_idx': (6, 9), 'cv_scores': array([ 0.92307692,  0.88      ,  1.        ,  0.95833333,  0.91666667]), 'avg_score': 0.93561538461538463, 'feature_names': ('6', '9')}, 3: {'feature_idx': (6, 9, 12), 'cv_scores': array([ 0.92307692,  0.92      ,  0.96      ,  1.        ,  0.95833333]), 'avg_score': 0.95228205128205123, 'feature_names': ('6', '9', '12')}, 4: {'feature_idx': (3, 6, 9, 12), 'cv_scores': array([ 0.96153846,  0.96      ,  0.96      ,  1.        ,  0.95833333]), 'avg_score': 0.96797435897435891, 'feature_names': ('3', '6', '9', '12')}, 5: {'feature_idx': (3, 6, 9, 10, 12), 'cv_scores': array([ 0.92307692,  0.96      ,  1.        ,  1.        ,  1.        ]), 'avg_score': 0.97661538461538466, 'feature_names': ('3', '6', '9', '10', '12')}, 6: {'feature_idx': (2, 3, 6, 9, 10, 12), 'cv_scores': array([ 0.92307692,  0.96      ,  1.        ,  0.95833333,  1.        ]), 'avg_score': 0.96828205128205125, 'feature_names': ('2', '3', '6', '9', '10', '12')}, 7: {'feature_idx': (0, 2, 3, 6, 9, 10, 12), 'cv_scores': array([ 0.92307692,  0.92      ,  1.        ,  1.        ,  1.        ]), 'avg_score': 0.96861538461538466, 'feature_names': ('0', '2', '3', '6', '9', '10', '12')}, 8: {'feature_idx': (0, 2, 3, 6, 8, 9, 10, 12), 'cv_scores': array([ 1.  ,  0.92,  1.  ,  1.  ,  1.  ]), 'avg_score': 0.98399999999999999, 'feature_names': ('0', '2', '3', '6', '8', '9', '10', '12')}, 9: {'feature_idx': (0, 2, 3, 6, 8, 9, 10, 11, 12), 'cv_scores': array([ 1.  ,  0.92,  1.  ,  1.  ,  1.  ]), 'avg_score': 0.98399999999999999, 'feature_names': ('0', '2', '3', '6', '8', '9', '10', '11', '12')}, 10: {'feature_idx': (0, 1, 2, 3, 6, 8, 9, 10, 11, 12), 'cv_scores': array([ 1.  ,  0.96,  1.  ,  1.  ,  1.  ]), 'avg_score': 0.99199999999999999, 'feature_names': ('0', '1', '2', '3', '6', '8', '9', '10', '11', '12')}}



![png](SequentialFeatureSelector_files/SequentialFeatureSelector_77_1.png)


## Example 10 -- Using other cross-validation schemes

In addition to standard k-fold and stratified k-fold, other cross validation schemes can be used with `SequentialFeatureSelector`. For example, `GroupKFold` or `LeaveOneOut` cross-validation from scikit-learn. 

#### Using GroupKFold with SequentialFeatureSelector


```python
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.data import iris_data
from sklearn.model_selection import GroupKFold
import numpy as np

X, y = iris_data()
groups = np.arange(len(y)) // 10
print('groups: {}'.format(groups))
```

    groups: [ 0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  2  2  2  2  2
      2  2  2  2  2  3  3  3  3  3  3  3  3  3  3  4  4  4  4  4  4  4  4  4  4
      5  5  5  5  5  5  5  5  5  5  6  6  6  6  6  6  6  6  6  6  7  7  7  7  7
      7  7  7  7  7  8  8  8  8  8  8  8  8  8  8  9  9  9  9  9  9  9  9  9  9
     10 10 10 10 10 10 10 10 10 10 11 11 11 11 11 11 11 11 11 11 12 12 12 12 12
     12 12 12 12 12 13 13 13 13 13 13 13 13 13 13 14 14 14 14 14 14 14 14 14 14]


Calling the `split()` method of a scikit-learn cross-validator object will return a generator that yields train, test splits.


```python
cv_gen = GroupKFold(4).split(X, y, groups)
cv_gen
```




    <generator object _BaseKFold.split at 0x1a1a041200>



The `cv` parameter of `SequentialFeatureSelector` must be either an `int` or an iterable yielding train, test splits. This iterable can be constructed by passing the train, test split generator to the built-in `list()` function. 


```python
cv = list(cv_gen)
```


```python
knn = KNeighborsClassifier(n_neighbors=2)
sfs = SFS(estimator=knn, 
          k_features=2,
          scoring='accuracy',
          cv=cv)

sfs.fit(X, y)

print('best combination (ACC: %.3f): %s\n' % (sfs.k_score_, sfs.k_feature_idx_))
```

    best combination (ACC: 0.940): (2, 3)
    


## Example 11 - Working with pandas DataFrames

## Example 12 - Using Pandas DataFrames

Optionally, we can also use pandas DataFrames and pandas Series as input to the `fit` function. In this case, the column names of the pandas DataFrame will be used as feature names. However, note that if `custom_feature_names` are provided in the fit function, these `custom_feature_names` take precedence over the DataFrame column-based feature names.


```python
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import SequentialFeatureSelector as SFS


iris = load_iris()
X = iris.data
y = iris.target
knn = KNeighborsClassifier(n_neighbors=4)

sfs1 = SFS(knn, 
           k_features=3, 
           forward=True, 
           floating=False, 
           scoring='accuracy',
           cv=0)
```


```python
X_df = pd.DataFrame(X, columns=['sepal len', 'petal len',
                                'sepal width', 'petal width'])
X_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal len</th>
      <th>petal len</th>
      <th>sepal width</th>
      <th>petal width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
    </tr>
  </tbody>
</table>
</div>



Also, the target array, `y`, can be optionally be cast as a Series:


```python
y_series = pd.Series(y)
y_series.head()
```




    0    0
    1    0
    2    0
    3    0
    4    0
    dtype: int64




```python
sfs1 = sfs1.fit(X_df, y_series)
```

Note that the only difference of passing a pandas DataFrame as input is that the sfs1.subsets_ array will now contain a new column, 


```python
sfs1.subsets_
```




    {1: {'avg_score': 0.95999999999999996,
      'cv_scores': array([ 0.96]),
      'feature_idx': (3,),
      'feature_names': ('petal width',)},
     2: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (2, 3),
      'feature_names': ('sepal width', 'petal width')},
     3: {'avg_score': 0.97333333333333338,
      'cv_scores': array([ 0.97333333]),
      'feature_idx': (1, 2, 3),
      'feature_names': ('petal len', 'sepal width', 'petal width')}}



In mlxtend version >= 0.13 pandas DataFrames are supported as feature inputs to the `SequentianFeatureSelector` instead of NumPy arrays or other NumPy-like array types.

# API


*SequentialFeatureSelector(estimator, k_features=1, forward=True, floating=False, verbose=0, scoring=None, cv=5, n_jobs=1, pre_dispatch='2*n_jobs', clone_estimator=True)*

Sequential Feature Selection for Classification and Regression.

**Parameters**

- `estimator` : scikit-learn classifier or regressor


- `k_features` : int or tuple or str (default: 1)

    Number of features to select,
    where k_features < the full feature set.
    New in 0.4.2: A tuple containing a min and max value can be provided,
    and the SFS will consider return any feature combination between
    min and max that scored highest in cross-validtion. For example,
    the tuple (1, 4) will return any combination from
    1 up to 4 features instead of a fixed number of features k.
    New in 0.8.0: A string argument "best" or "parsimonious".
    If "best" is provided, the feature selector will return the
    feature subset with the best cross-validation performance.
    If "parsimonious" is provided as an argument, the smallest
    feature subset that is within one standard error of the
    cross-validation performance will be selected.

- `forward` : bool (default: True)

    Forward selection if True,
    backward selection otherwise

- `floating` : bool (default: False)

    Adds a conditional exclusion/inclusion if True.

- `verbose` : int (default: 0), level of verbosity to use in logging.

    If 0, no output,
    if 1 number of features in current set, if 2 detailed logging i
    ncluding timestamp and cv scores at step.

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

- `cv` : int (default: 5)

    Integer or iterable yielding train, test splits. If cv is an integer
    and `estimator` is a classifier (or y consists of integer class
    labels) stratified k-fold. Otherwise regular k-fold cross-validation
    is performed. No cross-validation if cv is None, False, or 0.

- `n_jobs` : int (default: 1)

    The number of CPUs to use for evaluating different feature subsets
    in parallel. -1 means 'all CPUs'.

- `pre_dispatch` : int, or string (default: '2*n_jobs')

    Controls the number of jobs that get dispatched
    during parallel execution if `n_jobs > 1` or `n_jobs=-1`.
    Reducing this number can be useful to avoid an explosion of
    memory consumption when more jobs get dispatched than CPUs can process.
    This parameter can be:
    None, in which case all the jobs are immediately created and spawned.
    Use this for lightweight and fast-running jobs,
    to avoid delays due to on-demand spawning of the jobs
    An int, giving the exact number of total jobs that are spawned
    A string, giving an expression as a function
    of n_jobs, as in `2*n_jobs`

- `clone_estimator` : bool (default: True)

    Clones estimator if True; works with the original estimator instance
    if False. Set to False if the estimator doesn't
    implement scikit-learn's set_params and get_params methods.
    In addition, it is required to set cv=0, and n_jobs=1.

**Attributes**

- `k_feature_idx_` : array-like, shape = [n_predictions]

    Feature Indices of the selected feature subsets.

- `k_feature_names_` : array-like, shape = [n_predictions]

    Feature names of the selected feature subsets. If pandas
    DataFrames are used in the `fit` method, the feature
    names correspond to the column names. Otherwise, the
    feature names are string representation of the feature
    array indices. New in v 0.13.0.

- `k_score_` : float

    Cross validation average score of the selected subset.

- `subsets_` : dict

    A dictionary of selected feature subsets during the
    sequential selection, where the dictionary keys are
    the lengths k of these feature subsets. The dictionary
    values are dictionaries themselves with the following
    keys: 'feature_idx' (tuple of indices of the feature subset)
    'feature_names' (tuple of feature names of the feat. subset)
    'cv_scores' (list individual cross-validation scores)
    'avg_score' (average cross-validation score)
    Note that if pandas
    DataFrames are used in the `fit` method, the 'feature_names'
    correspond to the column names. Otherwise, the
    feature names are string representation of the feature
    array indices. The 'feature_names' is new in v 0.13.0.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/)

### Methods

<hr>

*fit(X, y, custom_feature_names=None, **fit_params)*

Perform feature selection and learn model from training data.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

- `y` : array-like, shape = [n_samples]

    Target values.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for y.

- `custom_feature_names` : None or tuple (default: tuple)

    Custom feature names for `self.k_feature_names` and
    `self.subsets_[i]['feature_names']`.
    (new in v 0.13.0)

- `fit_params` : dict of string -> object, optional

    Parameters to pass to to the fit method of classifier.

**Returns**

- `self` : object


<hr>

*fit_transform(X, y, **fit_params)*

Fit to training data then reduce X to its most important features.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

- `y` : array-like, shape = [n_samples]

    Target values.
    New in v 0.13.0: a pandas Series are now also accepted as
    argument for y.

- `fit_params` : dict of string -> object, optional

    Parameters to pass to to the fit method of classifier.

**Returns**

Reduced feature subset of X, shape={n_samples, k_features}

<hr>

*get_metric_dict(confidence_interval=0.95)*

Return metric dictionary

**Parameters**

- `confidence_interval` : float (default: 0.95)

    A positive float between 0.0 and 1.0 to compute the confidence
    interval bounds of the CV score averages.

**Returns**

Dictionary with items where each dictionary value is a list
    with the number of iterations (number of feature subsets) as
    its length. The dictionary keys corresponding to these lists
    are as follows:
    'feature_idx': tuple of the indices of the feature subset
    'cv_scores': list with individual CV scores
    'avg_score': of CV average scores
    'std_dev': standard deviation of the CV score average
    'std_err': standard error of the CV score average
    'ci_bound': confidence interval bound of the CV score average

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X)*

Reduce X to its most important features.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.
    New in v 0.13.0: pandas DataFrames are now also accepted as
    argument for X.

**Returns**

Reduced feature subset of X, shape={n_samples, k_features}


