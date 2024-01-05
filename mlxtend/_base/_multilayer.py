# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# Base MultiLayer (MultiLayer Parent Class)
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np


class _MultiLayer(object):
    def __init__(self):
        pass

    def _layermapping(self, n_features, n_classes, hidden_layers):
        """Creates a dictionaries of layer dimensions for weights and biases.

        For example, given
        `n_features=10`, `n_classes=10`, and `hidden_layers=[8, 7, 6]`:

        biases =
           {1: [[8], 'n_hidden_1'],
            2: [[7], 'n_hidden_2'],
            3: [[6], 'n_hidden_3'],
           'out': [[10], 'n_classes']
           }

        weights =
           {1: [[10, 8], 'n_features, n_hidden_1'],
            2: [[8, 7], 'n_hidden_1, n_hidden_2'],
            3: [[7, 6], 'n_hidden_2, n_hidden_3'],
            'out': [[6, 10], 'n_hidden_3, n_classes']
            }

        """
        weights = {
            "1": [[n_features, hidden_layers[0]], "n_features, n_hidden_1"],
            "out": [
                [hidden_layers[-1], n_classes],
                "n_hidden_%d, n_classes" % len(hidden_layers),
            ],
        }
        biases = {
            "1": [[hidden_layers[0]], "n_hidden_1"],
            "out": [[n_classes], "n_classes"],
        }

        if len(hidden_layers) > 1:
            for i, h in enumerate(hidden_layers[1:]):
                layer = i + 2
                weights[str(layer)] = [
                    [weights[str(layer - 1)][0][1], h],
                    "n_hidden_%d, n_hidden_%d" % (layer - 1, layer),
                ]
                biases[str(layer)] = [[h], "n_hidden_%d" % layer]
        return weights, biases

    def _init_params_from_layermapping(self, weight_maps, bias_maps, random_seed=None):
        rgen = np.random.RandomState(random_seed)
        weights, biases = {}, {}

        rgen = np.random.RandomState(random_seed)
        weights, biases = {}, {}

        for kw, kb in zip(sorted(weight_maps), sorted(bias_maps)):
            weights[kw] = rgen.normal(loc=0.0, scale=0.01, size=weight_maps[kw][0])
            biases[kb] = np.zeros(shape=bias_maps[kb][0])
        return weights, biases
