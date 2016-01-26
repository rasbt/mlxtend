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