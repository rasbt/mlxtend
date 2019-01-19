mlxtend version: 0.15.0dev 
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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/autompg_data/](http://rasbt.github.io/mlxtend/user_guide/data/autompg_data/)




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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/iris_data/](http://rasbt.github.io/mlxtend/user_guide/data/iris_data/)




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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/](http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/)




## make_multiplexer_dataset

*make_multiplexer_dataset(address_bits=2, sample_size=100, positive_class_ratio=0.5, shuffle=False, random_seed=None)*

Function to create a binary n-bit multiplexer dataset.

New in mlxtend v0.9

**Parameters**

- `address_bits` : int (default: 2)

    A positive integer that determines the number of address
    bits in the multiplexer, which in turn determine the
    n-bit capacity of the multiplexer and therefore the
    number of features. The number of features is determined by
    the number of address bits. For example, 2 address bits
    will result in a 6 bit multiplexer and consequently
    6 features (2 + 2^2 = 6). If `address_bits=3`, then
    this results in an 11-bit multiplexer as (2 + 2^3 = 11)
    with 11 features.


- `sample_size` : int (default: 100)

    The total number of samples generated.


- `positive_class_ratio` : float (default: 0.5)

    The fraction (a float between 0 and 1)
    of samples in the `sample_size`d dataset
    that have class label 1.
    If `positive_class_ratio=0.5` (default), then
    the ratio of class 0 and class 1 samples is perfectly balanced.


- `shuffle` : Bool (default: False)

    Whether or not to shuffle the features and labels.
    If `False` (default), the samples are returned in sorted
    order starting with `sample_size`/2 samples with class label 0
    and followed by `sample_size`/2 samples with class label 1.


- `random_seed` : int (default: None)

    Random seed used for generating the multiplexer samples and shuffling.

**Returns**

- `X, y` : [n_samples, n_features], [n_class_labels]

    X is the feature matrix with the number of samples equal
    to `sample_size`. The number of features is determined by
    the number of address bits. For instance, 2 address bits
    will result in a 6 bit multiplexer and consequently
    6 features (2 + 2^2 = 6).
    All features are binary (values in {0, 1}).
    y is a 1-dimensional array of class labels in {0, 1}.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/make_multiplexer_dataset](http://rasbt.github.io/mlxtend/user_guide/data/make_multiplexer_dataset)




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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/mnist_data/](http://rasbt.github.io/mlxtend/user_guide/data/mnist_data/)




## three_blobs_data

*three_blobs_data()*

A random dataset of 3 2D blobs for clustering.


- `Number of samples` : 150


- `Suggested labels` : {0, 1, 2}, distribution: [50, 50, 50]


**Returns**

- `X, y` : [n_samples, n_features], [n_cluster_labels]

    X is the feature matrix with 159 samples as rows
    and 2 feature columns.
    y is a 1-dimensional array of the 3 suggested cluster labels 0, 1, 2

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/three_blobs_data](http://rasbt.github.io/mlxtend/user_guide/data/three_blobs_data)




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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/data/wine_data](http://rasbt.github.io/mlxtend/user_guide/data/wine_data)




