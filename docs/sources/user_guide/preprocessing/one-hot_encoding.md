# One-Hot Encoding

A function that performs one-hot encoding for class labels.

> from mlxtend.preprocessing import one_hot

# Overview

Typical supervised machine learning algorithms for classifications assume that the class labels are *nominal* (a special case of *categorical* where no order is implied). A typical example of an nominal feature would be "color" since we can't say (in most applications) that "orange > blue > red".

The `one_hot` function provides a simple interface to convert class label integers into a so-called one-hot array, where each unique label is represented as a column in the new array.

For example, let's assume we have 5 data points from 3 different classes: 0, 1, and 2.

    y = [0, # sample 1, class 0 
         1, # sample 2, class 1 
         0, # sample 3, class 0
         2, # sample 4, class 2
         2] # sample 5, class 2
 
After one-hot encoding, we then obtain the following array (note that the index position of the "1" in each row denotes the class label of this sample):

    y = [[1,  0,  0], # sample 1, class 0 
         [0,  1,  0], # sample 2, class 1  
         [1,  0,  0], # sample 3, class 0
         [0,  0,  1], # sample 4, class 2
         [0,  0,  1]  # sample 5, class 2
         ]) 

# Examples

## Example 1 - Defaults


```python
from mlxtend.preprocessing import one_hot
import numpy as np

y = np.array([0, 1, 2, 1, 2])
one_hot(y)
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])



## Example 2 - Python Lists


```python
from mlxtend.preprocessing import one_hot

y = [0, 1, 2, 1, 2]
one_hot(y)
```




    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])



## Example 3 - Integer Arrays


```python
from mlxtend.preprocessing import one_hot

y = [0, 1, 2, 1, 2]
one_hot(y, dtype='int')
```




    array([[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [0, 1, 0],
           [0, 0, 1]])



## Example 4 - Arbitrary Numbers of Class Labels


```python
from mlxtend.preprocessing import one_hot

y = [0, 1, 2, 1, 2]
one_hot(y, num_labels=10)
```




    array([[ 1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])



# API


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

- `onehot` : numpy.ndarray, shape = [n_classlabels]

    One-hot encoded array, where each sample is represented as
    a row vector in the returned array.


