# Boston Housing Data

A function that loads the `boston_housing_data` dataset into NumPy arrays.

> from mlxtend.data import boston_housing_data

## Overview

The Boston Housing dataset for regression analysis.

**Features**

1. CRIM:      per capita crime rate by town
2. ZN:        proportion of residential land zoned for lots over 25,000 sq.ft.
3. INDUS:     proportion of non-retail business acres per town
4. CHAS:      Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
5. NOX:       nitric oxides concentration (parts per 10 million)
6. RM:        average number of rooms per dwelling
7. AGE:       proportion of owner-occupied units built prior to 1940
8. DIS:       weighted distances to five Boston employment centres
9. RAD:       index of accessibility to radial highways
10. TAX:      full-value property-tax rate per $10,000
11. PTRATIO:  pupil-teacher ratio by town
12. B:        1000(Bk - 0.63)^2 where Bk is the proportion of b. by town
13. LSTAT:    % lower status of the population
    

- Number of samples: 506

- Target variable (continuous): MEDV, Median value of owner-occupied homes in $1000's


### References

- Source: [https://archive.ics.uci.edu/ml/datasets/Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
- Harrison, D. and Rubinfeld, D.L. 
'Hedonic prices and the demand for clean air', J. Environ. Economics & Management, vol.5, 81-102, 1978.

## Example 1 - Dataset overview


```python
from mlxtend.data import boston_housing_data
X, y = boston_housing_data()

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('1st row', X[0])
```

    (506, 14)
    Dimensions: 506 x 13
    1st row [  6.32000000e-03   1.80000000e+01   2.31000000e+00   0.00000000e+00
       5.38000000e-01   6.57500000e+00   6.52000000e+01   4.09000000e+00
       1.00000000e+00   2.96000000e+02   1.53000000e+01   3.96900000e+02
       4.98000000e+00]


## API


*boston_housing_data()*

Boston Housing dataset.


- `Source` : https://archive.ics.uci.edu/ml/datasets/Housing


- `Number of samples` : 506



- `Continuous target variable` : MEDV

    MEDV = Median value of owner-occupied homes in $1000's

    Dataset Attributes:

    - 1) CRIM      per capita crime rate by town
    - 2) ZN        proportion of residential land zoned for lots over
    25,000 sq.ft.
    - 3) INDUS     proportion of non-retail business acres per town
    - 4) CHAS      Charles River dummy variable (= 1 if tract bounds
    river; 0 otherwise)
    - 5) NOX       nitric oxides concentration (parts per 10 million)
    - 6) RM        average number of rooms per dwelling
    - 7) AGE       proportion of owner-occupied units built prior to 1940
    - 8) DIS       weighted distances to five Boston employment centres
    - 9) RAD       index of accessibility to radial highways
    - 10) TAX      full-value property-tax rate per $10,000
    - 11) PTRATIO  pupil-teacher ratio by town
    - 12) B        1000(Bk - 0.63)^2 where Bk is the prop. of b. by town
    - 13) LSTAT    % lower status of the population

**Returns**

- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 506 housing samples as rows
    and 13 feature columns.
    y is a 1-dimensional array of the continuous target variable MEDV

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/boston_housing_data/](http://rasbt.github.io/mlxtend/user_guide/data/boston_housing_data/)


