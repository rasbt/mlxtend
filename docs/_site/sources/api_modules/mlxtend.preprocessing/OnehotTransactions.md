## OnehotTransactions

*OnehotTransactions(*args, **kwargs)*

Encoder class for transaction data in Python lists

**Parameters**

None

**Attributes**

columns_: list
List of unique names in the `X` input list of lists

**Examples**

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/)

### Methods

<hr>

*fit(X)*

Learn unique column names from transaction DataFrame

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

<hr>

*fit_transform(X, sparse=False)*

Fit a TransactionEncoder encoder and transform a dataset.

<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*inverse_transform(array)*

Transforms an encoded NumPy array back into transactions.

**Parameters**

- `array` : NumPy array [n_transactions, n_unique_items]

    The NumPy one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order

    For example,
```
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
```
    The corresponding column labels are available as self.columns_,
    e.g., ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

**Returns**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
```
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]
```

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X, sparse=False)*

Transform transactions into a one-hot encoded NumPy array.

**Parameters**

- `X` : list of lists

    A python list of lists, where the outer list stores the
    n transactions and the inner list stores the items in each
    transaction.

    For example,
    [['Apple', 'Beer', 'Rice', 'Chicken'],
    ['Apple', 'Beer', 'Rice'],
    ['Apple', 'Beer'],
    ['Apple', 'Bananas'],
    ['Milk', 'Beer', 'Rice', 'Chicken'],
    ['Milk', 'Beer', 'Rice'],
    ['Milk', 'Beer'],
    ['Apple', 'Bananas']]

    sparse: bool (default=False)
    If True, transform will return Compressed Sparse Row matrix
    instead of the regular one.

**Returns**

- `array` : NumPy array [n_transactions, n_unique_items]

    if sparse=False (default).
    Compressed Sparse Row matrix otherwise
    The one-hot encoded boolean array of the input transactions,
    where the columns represent the unique items found in the input
    array in alphabetic order. Exact representation depends
    on the sparse argument

    For example,
    array([[True , False, True , True , False, True ],
    [True , False, True , False, False, True ],
    [True , False, True , False, False, False],
    [True , True , False, False, False, False],
    [False, False, True , True , True , True ],
    [False, False, True , False, True , True ],
    [False, False, True , False, True , False],
    [True , True , False, False, False, False]])
    The corresponding column labels are available as self.columns_, e.g.,
    ['Apple', 'Bananas', 'Beer', 'Chicken', 'Milk', 'Rice']

