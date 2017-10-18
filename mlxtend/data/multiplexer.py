# Sebastian Raschka 2014-2017
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
