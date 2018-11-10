# ColumnSelector

Implementation of a column selector class for scikit-learn pipelines.

> from mlxtend.feature_selection import ColumnSelector

## Overview

The `ColumnSelector` can be used for "manual" feature selection, e.g., as part of a grid search via a scikit-learn pipeline.

### References

-

## Example 1 - Fitting an Estimator on a Feature Subset

Load a simple benchmark dataset:


```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
```

The `ColumnSelector` is a simple transformer class that selects specific columns (features) from a datast. For instance, using the `transform` method returns a reduced dataset that only contains two features (here: the first two features via the indices 0 and 1, respectively):


```python
from mlxtend.feature_selection import ColumnSelector

col_selector = ColumnSelector(cols=(0, 1))
# col_selector.fit(X) # optional, does not do anything
col_selector.transform(X).shape
```




    (150, 2)



`ColumnSelector` works both with numpy arrays and pandas dataframes:


```python
import pandas as pd

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df.head()
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
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




```python
col_selector = ColumnSelector(cols=("sepal length (cm)", "sepal width (cm)"))
col_selector.transform(iris_df).shape
```




    (150, 2)



Similarly, we can use the `ColumnSelector` as part of a scikit-learn `Pipeline`:


```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


pipe = make_pipeline(StandardScaler(),
                     ColumnSelector(cols=(0, 1)),
                     KNeighborsClassifier())

pipe.fit(X, y)
pipe.score(X, y)
```




    0.83999999999999997



## Example 2 - Feature Selection via GridSearch

Example 1 showed a simple useage example of the `ColumnSelector`; however, selecting columns from a dataset is trivial and does not require a specific transformer class since we could have achieved the same results via

```python
classifier.fit(X[:, :2], y)
classifier.score(X[:, :2], y)
```

However, the `ColumnSelector` becomes really useful for feature selection as part of a grid search as shown in this example.

Load a simple benchmark dataset:


```python
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
```

Create all possible combinations:


```python
from itertools import combinations

all_comb = []
for size in range(1, 5):
    all_comb += list(combinations(range(X.shape[1]), r=size))
print(all_comb)
```

    [(0,), (1,), (2,), (3,), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3), (0, 1, 2, 3)]


Feature and model selection via grid search:


```python
from mlxtend.feature_selection import ColumnSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(),
                     ColumnSelector(),
                     KNeighborsClassifier())

param_grid = {'columnselector__cols': all_comb,
              'kneighborsclassifier__n_neighbors': list(range(1, 11))}

grid = GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)
grid.fit(X, y)
print('Best parameters:', grid.best_params_)
print('Best performance:', grid.best_score_)
```

    Best parameters: {'columnselector__cols': (2, 3), 'kneighborsclassifier__n_neighbors': 1}
    Best performance: 0.98


## API


*ColumnSelector(cols=None, drop_axis=False)*

Object for selecting specific columns from a data set.

**Parameters**

- `cols` : array-like (default: None)

    A list specifying the feature indices to be selected. For example,
    [1, 4, 5] to select the 2nd, 5th, and 6th feature columns.
    If None, returns all columns in the array.


- `drop_axis` : bool (default=False)

    Drops last axis if True and the only one column is selected. This
    is useful, e.g., when the ColumnSelector is used for selecting
    only one column and the resulting array should be fed to e.g.,
    a scikit-learn column selector. E.g., instead of returning an
    array with shape (n_samples, 1), drop_axis=True will return an
    aray with shape (n_samples,).

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ColumnSelector/)

### Methods

<hr>

*fit(X, y=None)*

Mock method. Does nothing.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

self

<hr>

*fit_transform(X, y=None)*

Return a slice of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_slice` : shape = [n_samples, k_features]

    Subset of the feature space where k_features <= n_features

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

*transform(X, y=None)*

Return a slice of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_slice` : shape = [n_samples, k_features]

    Subset of the feature space where k_features <= n_features

<br><br>
