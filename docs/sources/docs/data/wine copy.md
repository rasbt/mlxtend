mlxtend  
Sebastian Raschka, last updated: 06/07/2015


<hr>

# Auto MPG

> from mlxtend.data import autompg_data

A function that loads the autompg dataset into NumPy arrays.

Source: [https://archive.ics.uci.edu/ml/datasets/Auto+MPG](https://archive.ics.uci.edu/ml/datasets/Auto+MPG)

The Auto-MPG dataset for regression analysis. The target (`y`) is defined as the miles per gallon (mpg) for 392 automobiles (6 rows containing "NaN"s have been removed. The 8 feature columns are:

1. cylinders: multi-valued discrete 
2. displacement: continuous 
3. horsepower: continuous 
4. weight: continuous 
5. acceleration: continuous 
6. model year: multi-valued discrete 
7. origin: multi-valued discrete 
8. car name: string (unique for each instance)

> Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.


<hr>

## Example

	>>> from mlxtend.data import autompg_data
    >>> X, y = autompg_data()
	
<hr>    

##Default Parameters


<pre>def autompg_data():
    """Auto MPG dataset.

    Returns
    --------
    X, y : [n_samples, n_features], [n_targets]
      X is the feature matrix with 392 auto samples as rows
      and 8 feature columns (6 rows with NaNs removed).
      y is a 1-dimensional array of the target MPG values.
      Source: https://archive.ics.uci.edu/ml/datasets/Auto+MPG

    """</pre>


