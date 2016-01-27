## iris_data

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

