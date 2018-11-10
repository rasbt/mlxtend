## cochrans_q

*cochrans_q(y_target, *y_model_predictions)*

Cochran's Q test to compare 2 or more models.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `*y_model_predictions` : array-likes, shape=[n_samples]

    Variable number of 2 or more arrays that
    contain the predicted class labels
    from models as 1D NumPy array.

**Returns**


- `q, p` : float or None, float

    Returns the Q (chi-squared) value and the p-value

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/](http://rasbt.github.io/mlxtend/user_guide/evaluate/cochrans_q/)

