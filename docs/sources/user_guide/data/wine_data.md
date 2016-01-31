# Wine Dataset

A function that loads the `Wine` dataset into NumPy arrays.

> from mlxtend.data import wine_data

# Overview

The Wine dataset for classification.

|				  |		  			|
|----------------------------|----------------|
| Samples                    | 178            |
| Features                   | 13             |
| Classes                    | 3              |
| Data Set Characteristics:  | Multivariate   |
| Attribute Characteristics: | Integer, Real  |
| Associated Tasks:          | Classification |
| Missing Values             | None           |

|	column| attribute	|
|-----|------------------------------|
| 1)  | Class Label                  |
| 2)  | Alcohol                      |
| 3)  | Malic acid                   |
| 4)  | Ash                          |
| 5)  | Alcalinity of ash            |
| 6)  | Magnesium                    |
| 7)  | Total phenols                |
| 8)  | Flavanoids                   |
| 9)  | Nonflavanoid phenols         |
| 10) | Proanthocyanins              |
| 11) | intensity                    |
| 12) | Hue                          |
| 13) | OD280/OD315 of diluted wines |
| 14) | Proline                      |


| class | samples   |
|-------|----|
| 0     | 59 |
| 1     | 71 |
| 2     | 48 |


### References

- Forina, M. et al, PARVUS - 
An Extendible Package for Data Exploration, Classification and Correlation. 
Institute of Pharmaceutical and Food Analysis and Technologies, Via Brigata Salerno, 
16147 Genoa, Italy. 
- Source: [https://archive.ics.uci.edu/ml/datasets/Wine](https://archive.ics.uci.edu/ml/datasets/Wine)
- Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

### Related Topics

- [Boston Housing Data](boston_housing.html)
- [Auto MPG](./autompg.html)
- [MNIST](./mnist.html)
- [Iris Dataset](./iris.html)

# Examples

## Example - Dataset overview


```python
from mlxtend.data import wine_data
X, y = wine_data()

print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
print('\nHeader: %s' % ['sepal length', 'sepal width',
                        'petal length', 'petal width'])
print('1st row', X[0])
```

    Dimensions: 178 x 13
    
    Header: ['sepal length', 'sepal width', 'petal length', 'petal width']
    1st row [  1.42300000e+01   1.71000000e+00   2.43000000e+00   1.56000000e+01
       1.27000000e+02   2.80000000e+00   3.06000000e+00   2.80000000e-01
       2.29000000e+00   5.64000000e+00   1.04000000e+00   3.92000000e+00
       1.06500000e+03]



```python
import numpy as np
print('Classes: %s' % np.unique(y))
print('Class distribution: %s' % np.bincount(y))
```

    Classes: [0 1 2]
    Class distribution: [59 71 48]


# API


*wine_data()*

Wine dataset.


- `Source` : https://archive.ics.uci.edu/ml/datasets/Wine


- `Number of samples` : 178


- `Class labels` : {0, 1, 2}, distribution: [59, 71, 48]


    Dataset Attributes:

    - 1) Alcohol
    - 2) Malic acid
    - 3) Ash
    - 4) Alcalinity of ash
    - 5) Magnesium
    - 6) Total phenols
    - 7) Flavanoids
    - 8) Nonflavanoid phenols
    - 9) Proanthocyanins
    - 10) Color intensity
    - 11) Hue
    - 12) OD280/OD315 of diluted wines
    - 13) Proline

**Returns**

- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 178 wine samples as rows
    and 13 feature columns.
    y is a 1-dimensional array of the 3 class labels 0, 1, 2


