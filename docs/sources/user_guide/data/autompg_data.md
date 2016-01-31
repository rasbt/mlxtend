# Auto MPG

A function that loads the `autompg` dataset into NumPy arrays.

> from mlxtend.data import autompg_data

# Overview

The Auto-MPG dataset for regression analysis. The target (`y`) is defined as the miles per gallon (mpg) for 392 automobiles (6 rows containing "NaN"s have been removed. The 8 feature columns are:

**Features**

1. cylinders: multi-valued discrete 
2. displacement: continuous 
3. horsepower: continuous 
4. weight: continuous 
5. acceleration: continuous 
6. model year: multi-valued discrete 
7. origin: multi-valued discrete 
8. car name: string (unique for each instance)

- Number of samples: 392

- Target variable (continuous): mpg


### References

- Source: [https://archive.ics.uci.edu/ml/datasets/Auto+MPG](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.

### Related Topics

- [Boston Housing Data](boston_housing.html)
- [Iris Dataset](./iris.html)
- [MNIST](./mnist.html)
- [Wine Dataset](./wine.html)

# Examples

## Example - Dataset overview


```python
from mlxtend.data import autompg_data
X, y = autompg_data()

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\nHeader: %s' % ['cylinders', 'displacement', 
                           'horsepower weight', 'acceleration',
                           'model year', 'origin', 'car name'])
print('1st row', X[0])
```

    Dimensions: 392 x 8
    
    Header: ['cylinders', 'displacement', 'horsepower weight', 'acceleration', 'model year', 'origin', 'car name']
    1st row ['8' '307.0' '130.0' '3504.0' '12.0' '70' '1' 'chevrolet chevelle malibu']


Note that the feature array contains a `str` column ("car name"), thus it is recommended to pick the features as needed and convert it into a `float` array for further analysis. The example below shows how to get rid of the `car name` column and cast the NumPy array as a `float` array.


```python
X[:, :-1].astype(float)
```




    array([[   8. ,  307. ,  130. , ...,   12. ,   70. ,    1. ],
           [   8. ,  350. ,  165. , ...,   11.5,   70. ,    1. ],
           [   8. ,  318. ,  150. , ...,   11. ,   70. ,    1. ],
           ..., 
           [   4. ,  135. ,   84. , ...,   11.6,   82. ,    1. ],
           [   4. ,  120. ,   79. , ...,   18.6,   82. ,    1. ],
           [   4. ,  119. ,   82. , ...,   19.4,   82. ,    1. ]])



# API


*autompg_data()*

Auto MPG dataset.



- `Source` : https://archive.ics.uci.edu/ml/datasets/Auto+MPG


- `Number of samples` : 392


- `Continuous target variable` : mpg


    Dataset Attributes:

    - 1) cylinders:  multi-valued discrete
    - 2) displacement: continuous
    - 3) horsepower: continuous
    - 4) weight: continuous
    - 5) acceleration: continuous
    - 6) model year: multi-valued discrete
    - 7) origin: multi-valued discrete
    - 8) car name: string (unique for each instance)

**Returns**

- `X, y` : [n_samples, n_features], [n_targets]

    X is the feature matrix with 392 auto samples as rows
    and 8 feature columns (6 rows with NaNs removed).
    y is a 1-dimensional array of the target MPG values.


