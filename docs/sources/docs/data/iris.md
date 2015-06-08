mlxtend  
Sebastian Raschka, last updated: 06/07/2015


<hr>

# Iris

> from mlxtend.data import iris_data

A function that loads the iris dataset into NumPy arrays.

Source:[https://archive.ics.uci.edu/ml/datasets/Iris](https://archive.ics.uci.edu/ml/datasets/Iris) 

> Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.

|iris.csv					  |		  			|
|----------------------------|----------------|
| Samples                    | 150            |
| Features                   | 4              |
| Classes                    | 3              |
| Data Set Characteristics:  | Multivariate   |
| Attribute Characteristics: | Real           |
| Associated Tasks:          | Classification |
| Missing Values             | None           |


|	column| attribute	|
|-----|------------------------------|
| 1)  | sepal length in cm                  |
| 2)  | sepal width in cm                      |
| 3)  | petal length in cm                   |
| 4)  | petal width in cm                        |
| 5)  | class label|


| class | samples   |
|-------|----|
| Iris-setosa     | 50 |
| Iris-versicolor     | 50 |
| Iris-virginica     | 50 |


Creator: R.A. Fisher (1936)

<hr>

## Example

	>>> from mlxtend.data import iris_data
    >>> X, y = iris_data()
	
<hr>    

##Default Parameters


<pre>def iris_data():
    """Iris flower dataset.

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
      X is the feature matrix with 150 flower samples as rows,
      and the 3 feature columns sepal length, sepal width,
      petal length, and petal width.
      y is a 1-dimensional array of the class labels where
      0 = setosa, 1 = versicolor, 2 = virginica.
      Reference: https://archive.ics.uci.edu/ml/datasets/Iris

     """</pre>


