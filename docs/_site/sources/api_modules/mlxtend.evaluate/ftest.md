## ftest

*ftest(y_target, *y_model_predictions)*

F-Test test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `f, p` : float or None, float

    Returns the F-value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/](http://rasbt.github.io/mlxtend/user_guide/evaluate/ftest/)

