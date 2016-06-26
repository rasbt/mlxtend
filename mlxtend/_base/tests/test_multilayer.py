# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend._base import _MultiLayer


class BlankClassifier(_MultiLayer):

    def __init__(self):
        pass


def test_layermapping():
    mlp = BlankClassifier()
    w, b = mlp._layermapping(n_features=10,
                             n_classes=11,
                             hidden_layers=[8, 7, 6])

    expect_b = {1: [[8], 'n_hidden_1'],
                2: [[7], 'n_hidden_2'],
                3: [[6], 'n_hidden_3'],
                'out': [[11], 'n_classes']}

    expect_w = {1: [[10, 8], 'n_features, n_hidden_1'],
                2: [[8, 7], 'n_hidden_1, n_hidden_2'],
                3: [[7, 6], 'n_hidden_2, n_hidden_3'],
                'out': [[6, 11], 'n_hidden_3, n_classes']}

    assert expect_b == b, b
    assert expect_w == w, w


def test_init_from_layermapping():
    mlp = BlankClassifier()
    wm, bm = mlp._layermapping(n_features=5,
                               n_classes=4,
                               hidden_layers=[3, 2])
    w, b = mlp._init_params_from_layermapping(weight_maps=wm,
                                              bias_maps=bm,
                                              random_seed=1)

    assert len(w.keys()) == 3
    assert len(b.keys()) == 3
    assert set(w.keys()) == set([1, 2, 'out'])
    assert set(b.keys()) == set([1, 2, 'out'])

    assert w[1].shape == (5, 3)
    assert w[2].shape == (3, 2)
    assert w['out'].shape == (2, 4)

    assert b[1].shape == (3,)
    assert b[2].shape == (2,)
    assert b['out'].shape == (4,)
