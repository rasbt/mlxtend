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

