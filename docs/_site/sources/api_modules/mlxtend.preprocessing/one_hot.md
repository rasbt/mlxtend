## one_hot

*one_hot(y, num_labels='auto', dtype='float')*

One-hot encoding of class labels

**Parameters**

- `y` : array-like, shape = [n_classlabels]

    Python list or numpy array consisting of class labels.

- `num_labels` : int or 'auto'

    Number of unique labels in the class label array. Infers the number
    of unique labels from the input array if set to 'auto'.

- `dtype` : str

    NumPy array type (float, float32, float64) of the output array.

**Returns**

- `ary` : numpy.ndarray, shape = [n_classlabels]

    One-hot encoded array, where each sample is represented as
    a row vector in the returned array.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/one_hot/)

