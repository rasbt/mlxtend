# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
#
# Implementation of a Multi-layer Perceptron in Tensorflow
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import tensorflow as tf
import numpy as np
from time import time
from .tf_base import _TfBaseClassifier


class TfMultiLayerPerceptron(_TfBaseClassifier):
    def __init__(self, eta=0.5, n_hidden=[50, 10],
                 activations=['softmax', 'softmax'],
                 epochs=100,
                 minibatches=1, random_seed=None,
                 print_progress=0, dtype=None):
        self.eta = eta
        if len(n_hidden) != len(activations):
            raise AttributeError('Number n_hidden and n_activations must be equal.')
        self.n_hidden = n_hidden
        self.activations = self._get_activations(activations)

        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress

        if dtype is None:
            self.dtype = tf.float32
        else:
            self.dtype = dtype

        return

    def _get_activations(self, activations):
        adict = {'softmax': tf.nn.sigmoid,
                 'relu': tf.nn.relu,
                 'tanh': tf.nn.tanh,
                 'elu': tf.nn.elu,
                 'softplus': tf.nn.softplus,
                 'softsign': tf.nn.softsign}
        act = {}
        for idx, a in enumerate(activations):
            if a not in adict:
                raise AttributeError('%s not in %s' % (a, list(adict.keys())))
            act[idx + 1] = adict[a]
        return act

    def fit(self, X, y, init_weights=True, override_minibatches=None):
        """

        """

        if override_minibatches:
            n_batches = override_minibatches
        else:
            n_batches = self.minibatches

        if y.shape[0] % n_batches != 0:
            raise AttributeError("Training set size %d cannot"
                                 " be divided into %d minibatches without"
                                 " remainder" % (y.shape[0], n_batches))

        # Construct the Graph
        g = tf.Graph()
        with g.as_default():

            if init_weights:
                self._n_classes = np.max(y) + 1
                self._n_features = X.shape[1]
                self._weight_maps, self._bias_maps = self._layermapping(
                                                         n_features=self._n_features,
                                                         n_classes=self._n_classes,
                                                         n_hidden=self.n_hidden)
                tf_weights, tf_biases = self._initialize_weights(
                                            weight_maps=self._weight_maps,
                                            bias_maps=self._bias_maps)
                self.cost_ = []
            else:
                tf_weights, tf_biases = self._reuse_weights(weights=self.weights_, baises=self.biases_)

            # Prepare the training data
            y_enc = self._one_hot(y, self._n_classes)
            n_idx = list(range(y.shape[0]))
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            tf_y = tf.convert_to_tensor(value=y_enc, dtype=self.dtype)

            tf_idx = tf.placeholder(tf.int32,
                                    shape=[int(y.shape[0] / n_batches)])
            X_batch = tf.gather(params=tf_X, indices=tf_idx)
            y_batch = tf.gather(params=tf_y, indices=tf_idx)

            # Setup the graph for minimizing cross entropy cost

            logits = self._predict(tf_X=tf_X, tf_weights=tf_weights, tf_biases=tf_biases, activations=self.activations)

            # Define loss and optimizer
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, tf_y)
            cost = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.eta)
            train = optimizer.minimize(cost)

            # Initializing the variables
            init = tf.initialize_all_variables()

        # random seed for shuffling
        if self.random_seed:
            np.random.seed(self.random_seed)

        # Launch the graph
        with tf.Session(graph=g) as sess:
            sess.run(init)
            self.init_time_ = time()
            for epoch in range(self.epochs):

                if self.minibatches > 1:
                    n_idx = np.random.permutation(n_idx)
                minis = np.array_split(n_idx, self.minibatches)
                costs = []
                for idx in minis:
                    _, c = sess.run([train, cost], feed_dict={tf_idx: idx})
                    costs.append(c)
                avg_cost = np.mean(costs)
                self.cost_.append(avg_cost)
                self._print_progress(epoch + 1, avg_cost)
            self.weights_ = {k: tf_weights[k].eval() for k in tf_weights}
            self.biases_ = {k: tf_biases[k].eval() for k in tf_biases}

        return

    def predict(self, X):
        """Predict class labels of X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class_labels : array-like, shape = [n_samples]
          Predicted class labels.

        """
        return np.argmax(self.predict_proba(X=X), axis=1)

    def predict_proba(self, X):
        """Predict class probabilities of X from the net input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        Class probabilties : array-like, shape= [n_samples, n_classes]

        """
        if not hasattr(self, 'weights_'):
            raise AttributeError('The model has not been fitted, yet.')

        with tf.Session():
            tf.initialize_all_variables().run()
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            logits = self._predict(tf_X=tf_X, tf_weights=self.weights_, tf_biases=self.biases_, activations=self.activations)
            return logits.eval()

    def _layermapping(self, n_features, n_classes, n_hidden):
        """Creates a dictionaries of layer dimensions for weights and biases.

        For example, given
        `n_features=10`, `n_classes=10`, and `n_hidden=[8, 7, 6]`:

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
        weights =  {1: [[n_features, n_hidden[0]], 'n_features, n_hidden_1'],
                    'out': [[n_hidden[-1], n_classes], 'n_hidden_%d, n_classes' % len(n_hidden)]}
        biases = {1: [[n_hidden[0]], 'n_hidden_1'],
                  'out': [[n_classes], 'n_classes']}

        if len(n_hidden) > 1:
            for i, h in enumerate(n_hidden[1:]):
                layer = i + 2
                weights[layer] = [[weights[layer - 1][0][1], h],
                                  'n_hidden_%d, n_hidden_%d' % (layer - 1, layer)]
                biases[layer] = [[h], 'n_hidden_%d' % layer]
        return weights, biases

    def _predict(self, tf_X, tf_weights, tf_biases, activations):
        hidden_1 = self.activations[1](tf.add(tf.matmul(tf_X, tf_weights[1]), tf_biases[1]))
        prev_layer = hidden_1
        if len(tf_weights) > 2:
            for layer in range(2, len(tf_weights)):
                prev_layer = self.activations[layer](tf.add(tf.matmul(prev_layer, tf_weights[layer]), tf_biases[layer]))
        logits = tf.matmul(prev_layer, tf_weights['out']) + tf_biases['out']
        return logits

    def _resuse_weights(self, weights, biases):
            w = {k: tf.Variable(self.weights_[k]) for k in self.weights_}
            b = {k: tf.Variable(self.biases_[k]) for k in self.biases_}
            return w, b

    def _initialize_weights(self, weight_maps, bias_maps):
        tf_weights, tf_biases = {}, {}
        for i, k in enumerate(zip(weight_maps, bias_maps)):
            if self.random_seed:
                seed = self.random_seed + i
            else:
                seed = None
            tf_weights[k[0]] = tf.Variable(tf.truncated_normal(weight_maps[k[0]][0],
                                                               seed=seed))
            tf_biases[k[1]] = tf.zeros(bias_maps[k[1]][0])
        return tf_weights, tf_biases

    def _one_hot(self, y, n_labels):
        mat = np.zeros((len(y), n_labels))
        for i, val in enumerate(y):
            mat[i, val] = 1
        return mat.astype(float)

    def _to_classlabels(self, z):
        return z.argmax(axis=1)
