# Standardize

A function that performs column-based standardization on a NumPy array.

> from mlxtend.preprocessing import standardize

# Overview

The result of standardization (or Z-score normalization) is that the features will be rescaled so that they'll have the properties of a standard normal distribution with

$\mu = 0$ and $\sigma = 1$.

where $\mu$ is the mean (average) and $\sigma$ is the standard deviation from the mean; standard scores (also called z scores) of the samples are calculated as

$$z=\frac{x-\mu}{\sigma}.$$

Standardizing the features so that they are centered around 0 with a standard deviation of 1 is not only important if we are comparing measurements that have different units, but it is also a general requirement for the optimal performance of many machine learning algorithms. 

One family of algorithms that is scale-invariant encompasses tree-based learning algorithms. Let's take the general CART decision tree algorithm. Without going into much depth regarding information gain and impurity measures, we can think of the decision as "is feature x_i >= some_val?" Intuitively, we can see that it really doesn't matter on which scale this feature is (centimeters, Fahrenheit, a standardized scale -- it really doesn't matter).


Some examples of algorithms where feature scaling matters are:


- k-nearest neighbors with an Euclidean distance measure if want all features to contribute equally
- k-means (see k-nearest neighbors)
- logistic regression, SVMs, perceptrons, neural networks etc. if you are using gradient descent/ascent-based optimization, otherwise some weights will update much faster than others
- linear discriminant analysis, principal component analysis, kernel principal component analysis since you want to find directions of maximizing the variance (under the constraints that those directions/eigenvectors/principal components are orthogonal); you want to have features on the same scale since you'd emphasize variables on "larger measurement scales" more.


There are many more cases than I can possibly list here ... I always recommend you to think about the algorithm and what it's doing, and then it typically becomes obvious whether we want to scale your features or not.


In addition, we'd also want to think about whether we want to "standardize" or "normalize" (here: scaling to [0, 1] range) our data. Some algorithms assume that our data is centered at 0. For example, if we initialize the weights of a small multi-layer perceptron with tanh activation units to 0 or small random values centered around zero, we want to update the model weights "equally."
As a rule of thumb I'd say: When in doubt, just standardize the data, it shouldn't hurt.   


 

### Related Topics

- [MeanCenterer](./mean_centering.md)
- [MinMaxScaling](./minmax_scaling.md)
- [Shuffle Arrays in Unison](./shuffle_arrays_unison.md)
- [DenseTransformer](./scikit-learn_dense_transformer.md)

# Examples

## Example 1 - Standardize a Pandas DataFrame


```python
import pandas as pd

s1 = pd.Series([1, 2, 3, 4, 5, 6], index=(range(6)))
s2 = pd.Series([10, 9, 8, 7, 6, 5], index=(range(6)))
df = pd.DataFrame(s1, columns=['s1'])
df['s2'] = s2
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>s1</th>
      <th>s2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>6</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
from mlxtend.preprocessing import standardize
standardize(df, columns=['s1', 's2'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>s1</th>
      <th>s2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1.46385</td>
      <td>1.46385</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.87831</td>
      <td>0.87831</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.29277</td>
      <td>0.29277</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.29277</td>
      <td>-0.29277</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.87831</td>
      <td>-0.87831</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.46385</td>
      <td>-1.46385</td>
    </tr>
  </tbody>
</table>
</div>



## Example 2 - Standardize a NumPy Array


```python
import numpy as np

X = np.array([[1, 10], [2, 9], [3, 8], [4, 7], [5, 6], [6, 5]])
X
```




    array([[ 1, 10],
           [ 2,  9],
           [ 3,  8],
           [ 4,  7],
           [ 5,  6],
           [ 6,  5]])




```python
from mlxtend.preprocessing import standardize
standardize(X, columns=[0, 1])
```




    array([[-1.46385011,  1.46385011],
           [-0.87831007,  0.87831007],
           [-0.29277002,  0.29277002],
           [ 0.29277002, -0.29277002],
           [ 0.87831007, -0.87831007],
           [ 1.46385011, -1.46385011]])



## Example 3 - Re-using parameters

In machine learning contexts, it is desired to re-use the parameters that have been obtained from a training set to scale new, future data (including the independent test set). By setting `return_params=True`, the `standardize` function returns a second object, a parameter dictionary containing the column means and standard deviations that can be re-used by feeding it to the `params` parameter upon function call.


```python
import numpy as np
from mlxtend.preprocessing import standardize

X_train = np.array([[1, 10], [4, 7], [3, 8]])
X_test = np.array([[1, 2], [3, 4], [5, 6]])

X_train_std, params = standardize(X_train, columns=[0, 1], return_params=True)
X_train_std
```




    array([[-1.33630621,  1.33630621],
           [ 1.06904497, -1.06904497],
           [ 0.26726124, -0.26726124]])




```python
params
```




    {'avgs': array([ 2.66666667,  8.33333333]),
     'stds': array([ 1.24721913,  1.24721913])}




```python
X_test_std = standardize(X_test, columns=[0, 1], params=params)
X_test_std
```




    array([[-1.33630621, -5.0779636 ],
           [ 0.26726124, -3.47439614],
           [ 1.87082869, -1.87082869]])



# API


*standardize(array, columns, ddof=0, return_params=False, params=None)*

Standardize columns in pandas DataFrames.

**Parameters**

- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].


- `columns` : array-like, shape = [n_columns]

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]

- `ddof` : int (default: 0)

    Delta Degrees of Freedom. The divisor used in calculations
    is N - ddof, where N represents the number of elements.

- `return_params` : dict (default: False)

    If set to True, a dictionary is returned in addition to the
    standardized array. The parameter dictionary contains the
    column means ('avgs') and standard deviations ('stds') of
    the individual columns.

- `params` : dict (default: None)

    A dictionary with column means and standard deviations as
    returned by the `standardize` function if `return_params`
    was set to True. If a `params` dictionary is provided, the
    `standardize` function will use these instead of computing
    them from the current array.

**Notes**

If all values in a given column are the same, these values are all
    set to `0.0`. The standard deviation in the `parameters` dictionary
    is consequently set to `1.0` to avoid dividing by zero.

**Returns**

- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with standardized columns.


