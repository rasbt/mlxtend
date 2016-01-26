## autompg_data



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

## boston_housing_data



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
    - 12) B        1000(Bk - 0.63)^2 where Bk is the proportion of b. by town
    - 13) LSTAT    % lower status of the population

**Returns**


- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 506 housing samples as rows
    and 13 feature columns.
    y is a 1-dimensional array of the continuous target variable MEDV

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

## loadlocal_mnist



*loadlocal_mnist(images_path, labels_path)*

Read MNIST from ubyte files.

**Parameters**


- `images_path` : str

    path to the test or train MNIST ubyte file

- `labels_path` : str

    path to the test or train MNIST class labels file

**Returns**


- `images` : [n_samples, n_pixels] numpy.array

    Pixel values of the images.

- `labels` : [n_samples] numpy array

    Target class labels

## mnist_data



*mnist_data()*

5000 samples from the MNIST handwritten digits dataset.


- `Data Source` : http://yann.lecun.com/exdb/mnist/


**Returns**


- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with 5000 image samples as rows,
    each row consists of 28x28 pixels that were unrolled into
    784 pixel feature vectors.
    y contains the 10 unique class labels 0-9.

## wine_data



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

