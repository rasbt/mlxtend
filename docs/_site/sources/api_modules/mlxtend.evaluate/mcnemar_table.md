## mcnemar_table

*mcnemar_table(y_target, y_model1, y_model2)*

Compute a 2x2 contigency table for McNemar's test.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.

- `y_model1` : array-like, shape=[n_samples]

    Predicted class labels from model as 1D NumPy array.

- `y_model2` : array-like, shape=[n_samples]

    Predicted class labels from model 2 as 1D NumPy array.

**Returns**

- `tb` : array-like, shape=[2, 2]

    2x2 contingency table with the following contents:
    a: tb[0, 0]: # of samples that both models predicted correctly
    b: tb[0, 1]: # of samples that model 1 got right and model 2 got wrong
    c: tb[1, 0]: # of samples that model 2 got right and model 1 got wrong
    d: tb[1, 1]: # of samples that both models predicted incorrectly

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_table/)

