# Iris Dataset

A function that loads the `iris` dataset into NumPy arrays.

> from mlxtend.data import iris_data

## Overview

The Iris dataset for classification.

**Features**

1. Sepal length
2. Sepal width
3. Petal length
4. Petal width


- Number of samples: 150


- Target variable (discrete): {50x Setosa, 50x Versicolor, 50x Virginica}


### References

- Source: [https://archive.ics.uci.edu/ml/datasets/Iris](https://archive.ics.uci.edu/ml/datasets/Iris) 
- Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.


## Example 1 - Dataset overview


```python
from mlxtend.data import iris_data
X, y = iris_data()

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\nHeader: %s' % ['sepal length', 'sepal width',
                        'petal length', 'petal width'])
print('1st row', X[0])
```

    Dimensions: 150 x 4
    
    Header: ['sepal length', 'sepal width', 'petal length', 'petal width']
    1st row [ 5.1  3.5  1.4  0.2]



```python
import numpy as np
print('Classes: Setosa, Versicolor, Virginica')
print(np.unique(y))
print('Class distribution: %s' % np.bincount(y))
```

    Classes: Setosa, Versicolor, Virginica
    [0 1 2]
    Class distribution: [50 50 50]


## API


*iris_data()*

Iris flower dataset.


- `Source` : https://archive.ics.uci.edu/ml/datasets/Iris


- `Number of samples` : 150


- `Class labels` : {0, 1, 2}, distribution: [50, 50, 50]

    0 = setosa, 1 = versicolor, 2 = virginica.

    Dataset Attributes:

    - 1) sepal length [cm]
    - 2) sepal width [cm]
    - 3) petal length [cm]
    - 4) petal width [cm]

**Returns**

- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 150 flower samples as rows,
    and 4 feature columns sepal length, sepal width,
    petal length, and petal width.
    y is a 1-dimensional array of the class labels {0, 1, 2}

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/iris_data/](http://rasbt.github.io/mlxtend/user_guide/data/iris_data/)


