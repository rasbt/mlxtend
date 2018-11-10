## mcnemar_tables

*mcnemar_tables(y_target, *y_model_predictions)*

Compute multiple 2x2 contigency tables for McNemar's
test or Cochran's Q test.

**Parameters**

- `y_target` : array-like, shape=[n_samples]

    True class labels as 1D NumPy array.


- `y_model_predictions` : array-like, shape=[n_samples]

    Predicted class labels for a model.

**Returns**


- `tables` : dict

    Dictionary of NumPy arrays with shape=[2, 2]. Each dictionary
    key names the two models to be compared based on the order the
    models were passed as `*y_model_predictions`. The number of
    dictionary entries is equal to the number of pairwise combinations
    between the m models, i.e., "m choose 2."

    For example the following target array (containing the true labels)
    and 3 models

    - y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    - y_mod0 = np.array([0, 1, 0, 0, 0, 1, 1, 0, 0, 0])
    - y_mod0 = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 0])
    - y_mod0 = np.array([0, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    would result in the following dictionary:


    {'model_0 vs model_1': array([[ 4.,  1.],
    [ 2.,  3.]]),
    'model_0 vs model_2': array([[ 3.,  0.],
    [ 3.,  4.]]),
    'model_1 vs model_2': array([[ 3.,  0.],
    [ 2.,  5.]])}

    Each array is structured in the following way:

    - tb[0, 0]: # of samples that both models predicted correctly
    - tb[0, 1]: # of samples that model a got right and model b got wrong
    - tb[1, 0]: # of samples that model b got right and model a got wrong
    - tb[1, 1]: # of samples that both models predicted incorrectly

**Examples**

    For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/](http://rasbt.github.io/mlxtend/user_guide/evaluate/mcnemar_tables/)

