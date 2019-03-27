# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
#
# A function for creating a multiplexer dataset for classification.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


def make_multiplexer_dataset(address_bits=2, sample_size=100,
                             positive_class_ratio=0.5, shuffle=False,
                             random_seed=None):
    """
    Function to create a binary n-bit multiplexer dataset.

    New in mlxtend v0.9

    Parameters
    ---------------
    address_bits : int (default: 2)
        A positive integer that determines the number of address
        bits in the multiplexer, which in turn determine the
        n-bit capacity of the multiplexer and therefore the
        number of features. The number of features is determined by
        the number of address bits. For example, 2 address bits
        will result in a 6 bit multiplexer and consequently
        6 features (2 + 2^2 = 6). If `address_bits=3`, then
        this results in an 11-bit multiplexer as (2 + 2^3 = 11)
        with 11 features.

    sample_size : int (default: 100)
        The total number of samples generated.

    positive_class_ratio : float (default: 0.5)
        The fraction (a float between 0 and 1)
        of samples in the `sample_size`d dataset
        that have class label 1.
        If `positive_class_ratio=0.5` (default), then
        the ratio of class 0 and class 1 samples is perfectly balanced.

    shuffle : Bool (default: False)
        Whether or not to shuffle the features and labels.
        If `False` (default), the samples are returned in sorted
        order starting with `sample_size`/2 samples with class label 0
        and followed by `sample_size`/2 samples with class label 1.

    random_seed : int (default: None)
        Random seed used for generating the multiplexer samples and shuffling.

    Returns
    --------
    X, y : [n_samples, n_features], [n_class_labels]
        X is the feature matrix with the number of samples equal
        to `sample_size`. The number of features is determined by
        the number of address bits. For instance, 2 address bits
        will result in a 6 bit multiplexer and consequently
        6 features (2 + 2^2 = 6).
        All features are binary (values in {0, 1}).
        y is a 1-dimensional array of class labels in {0, 1}.

    Examples
    -----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/data/make_multiplexer_dataset

    """

    if not isinstance(address_bits, int):
        raise AttributeError('address_bits'
                             ' must be an integer. Got %s.' %
                             type(address_bits))
    if address_bits < 1:
        raise AttributeError('Number of address_bits'
                             ' must be greater than 0. Got %s.' % address_bits)
    register_bits = 2**address_bits
    total_bits = address_bits + register_bits
    X_pos, y_pos = [], []
    X_neg, y_neg = [], []

    # use numpy's instead of python's round because of consistent
    # banker's rounding behavior across versions
    n_positives = np.round(sample_size*positive_class_ratio).astype(np.int)
    n_negatives = sample_size - n_positives

    rng = np.random.RandomState(random_seed)

    def gen_randsample():
        all_bits = [rng.randint(0, 2) for i in range(total_bits)]
        address_str = ''.join(str(c) for c in all_bits[:address_bits])
        register_pos = int(address_str, base=2)
        class_label = all_bits[address_bits:][register_pos]
        return all_bits, class_label

    while len(y_pos) < n_positives or len(y_neg) < n_negatives:

        all_bits, class_label = gen_randsample()

        if class_label and len(y_pos) < n_positives:
            X_pos.append(all_bits)
            y_pos.append(class_label)

        elif not class_label and len(y_neg) < n_negatives:
            X_neg.append(all_bits)
            y_neg.append(class_label)

    X, y = X_pos + X_neg, y_pos + y_neg
    X, y = np.array(X, dtype=np.int), np.array(y, dtype=np.int)

    if shuffle:
        p = rng.permutation(y.shape[0])
        X, y = X[p], y[p]

    return X, y
