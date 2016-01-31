# MinMax Scaling

A function for min-max scaling of pandas DataFrames or NumPy arrays.

> from mlxtend.preprocessing import MinMaxScaling

### Related Topics

- [Standardize](./standardize.md)
- [MeanCenterer](./mean_centerer.md)
- [Shuffle Arrays in Unison](./shuffle_arrays_unison.md)
- [DenseTransformer](./scikit-learn_dense_transformer.md)

An alternative approach to Z-score normalization (or standardization) is the so-called Min-Max scaling (often also simply called "normalization" - a common cause for ambiguities).
In this approach, the data is scaled to a fixed range - usually 0 to 1.
The cost of having this bounded range - in contrast to standardization - is that we will end up with smaller standard deviations, which can suppress the effect of outliers.

A Min-Max scaling is typically done via the following equation:

$$X_{sc} = \frac{X - X_{min}}{X_{max} - X_{min}}.$$

One family of algorithms that is scale-invariant encompasses tree-based learning algorithms. Let's take the general CART decision tree algorithm. Without going into much depth regarding information gain and impurity measures, we can think of the decision as "is feature x_i >= some_val?" Intuitively, we can see that it really doesn't matter on which scale this feature is (centimeters, Fahrenheit, a standardized scale -- it really doesn't matter).


Some examples of algorithms where feature scaling matters are:


- k-nearest neighbors with an Euclidean distance measure if want all features to contribute equally
- k-means (see k-nearest neighbors)
- logistic regression, SVMs, perceptrons, neural networks etc. if you are using gradient descent/ascent-based optimization, otherwise some weights will update much faster than others
- linear discriminant analysis, principal component analysis, kernel principal component analysis since you want to find directions of maximizing the variance (under the constraints that those directions/eigenvectors/principal components are orthogonal); you want to have features on the same scale since you'd emphasize variables on "larger measurement scales" more.


There are many more cases than I can possibly list here ... I always recommend you to think about the algorithm and what it's doing, and then it typically becomes obvious whether we want to scale your features or not.


In addition, we'd also want to think about whether we want to "standardize" or "normalize" (here: scaling to [0, 1] range) our data. Some algorithms assume that our data is centered at 0. For example, if we initialize the weights of a small multi-layer perceptron with tanh activation units to 0 or small random values centered around zero, we want to update the model weights "equally."
As a rule of thumb I'd say: When in doubt, just standardize the data, it shouldn't hurt.   


 

# Examples

## Example 1 - Scaling a Pandas DataFrame


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
from mlxtend.preprocessing import minmax_scaling
minmax_scaling(df, columns=['s1', 's2'])
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
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.2</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.4</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.8</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



## Example 2 - Scaling a NumPy Array


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
from mlxtend.preprocessing import minmax_scaling
minmax_scaling(X, columns=[0, 1])
```




    array([[ 0. ,  1. ],
           [ 0.2,  0.8],
           [ 0.4,  0.6],
           [ 0.6,  0.4],
           [ 0.8,  0.2],
           [ 1. ,  0. ]])



# API


*minmax_scaling(array, columns, min_val=0, max_val=1)*

Min max scaling of pandas' DataFrames.

**Parameters**

- `array` : pandas DataFrame or NumPy ndarray, shape = [n_rows, n_columns].


- `columns` : array-like, shape = [n_columns]

    Array-like with column names, e.g., ['col1', 'col2', ...]
    or column indices [0, 2, 4, ...]

- `min_val` : `int` or `float`, optional (default=`0`)

    minimum value after rescaling.

- `min_val` : `int` or `float`, optional (default=`1`)

    maximum value after rescaling.

**Returns**

- `df_new` : pandas DataFrame object.

    Copy of the array or DataFrame with rescaled columns.


