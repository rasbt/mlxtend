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
    """Multi-layer perceptron classifier.

    Parameters
    ------------
    eta : float (default: 0.5)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
    hidden_layers : list (default: [50, 10])
        Number of units per hidden layer. By default 50 units in the
        first hidden layer, and 10 hidden units in the second hidden layer.
    activations : list (default: ['logistic', 'logistic'])
        Activation functions for each layer.
        Available actiavtion functions:
        "logistic", "relu", "tanh", "relu6", "elu", "softplus", "softsign"
    optimizer : str (default: "gradientdescent")
        Optimizer to minimize the cost function:
        "gradientdescent", "momentum", "adam", "ftrl", "adagrad"
    momentum : float (default: 0.0)
        Momentum constant for momentum learning; only applies if
        optimizer='momentum'
    l1 : float (default: 0.0)
        L1 regularization strength; only applies if optimizer='ftrl'
    l2 : float (default: 0.0)
        regularization strength; only applies if optimizer='ftrl'
    minibatches : int (default: 1)
        Divide the training data into *k* minibatches
        for accelerated stochastic gradient descent learning.
        Gradient Descent Learning if `minibatches` = 1
        Stochastic Gradient Descent learning if `minibatches` = len(y)
        Minibatch learning if `minibatches` > 1
    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.
    print_progress : int (default: 0)
        Prints progress in fitting to stderr.
        0: No output
        1: Epochs elapsed and cost
        2: 1 plus time elapsed
        3: 2 plus estimated time until completion
    dtype : Array-type (default: None)
        Uses tensorflow.float32 if None.

    Attributes
    -----------
    weights_ : 2d-array, shape=[n_features, n_classes]
        Weights after fitting.
    biases_ : 1D-array, shape=[n_classes]
        Bias units after fitting.
    cost_ : list
        List of floats, the average cross_entropy for each epoch.

    """
    def __init__(self, eta=0.5, epochs=50,
                 hidden_layers=[50, 10],
                 activations=['logistic', 'logistic'],
                 optimizer='gradientdescent',
                 momentum=0.0, l1=0.0, l2=0.0,
                 minibatches=1, random_seed=None,
                 print_progress=0, dtype=None):
        self.eta = eta
        if len(hidden_layers) != len(activations):
            raise AttributeError('Number of hidden_layers and'
                                 ' n_activations must be equal.')
        self.hidden_layers = hidden_layers
        self.momentum = momentum
        self.activations = self._get_activations(activations)
        self.l1 = l1
        self.l2 = l2
        self.optimizer = self._init_optimizer(optimizer)
        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress

        if dtype is None:
            self.dtype = tf.float32
        else:
            self.dtype = dtype

        return

    def _init_optimizer(self, optimizer):
        if optimizer == 'gradientdescent':
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.eta)
        elif optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=self.eta,
                                             momentum=self.momentum)
        elif optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.eta)
        elif optimizer == 'ftrl':
            opt = tf.train.FtrlOptimizer(
                learning_rate=self.eta,
                l1_regularization_strength=self.l1,
                l2_regularization_strength=self.l2)
        elif optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.eta)
        else:
            raise AttributeError('optimizer must be "gradientdescent",'
                                 ' "momentum", "adam", "ftrl", or "adagrad"')
        return opt

    def _get_activations(self, activations):
        adict = {'logistic': tf.nn.sigmoid,
                 'relu': tf.nn.relu,
                 'relu6': tf.nn.relu6,
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

    def fit(self, X, y, init_weights=True,
            override_minibatches=None, n_classes=None,
            X_valid=None, y_valid=None):
        """Learn weight coefficients from training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        y : array-like, shape = [n_samples]
            Target values.
        init_weights : bool (default: True)
            (Re)initializes weights to small random floats if True.
        override_minibatches : int or None (default: None)
            Uses a different number of minibatches for this session.
        n_classes : int (default: None)
            A positive integer to declare the number of class labels
            if not all class labels are present in a partial training set.
            Gets the number of class labels automatically if None.
            Ignored if init_weights=False.
        X_valid : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Optional validation set to store the validation accuracy values
            for each epoch via self.valid_acc_
        y_valid : array-like, shape = [n_samples]
            Target values for X_valid

        Returns
        -------
        self : object

        """
        self._check_arrays(X, y)
        if override_minibatches:
            n_batches = override_minibatches
        else:
            n_batches = self.minibatches

        if y.shape[0] % n_batches != 0:
            raise AttributeError("Training set size %d cannot"
                                 " be divided into %d minibatches without"
                                 " remainder" % (y.shape[0], n_batches))

        if hasattr(X_valid, 'shape'):
            validation = True
        else:
            validation = False

        # Construct the Graph
        g = tf.Graph()
        with g.as_default():

            if init_weights:
                if n_classes:
                    self._n_classes = n_classes
                else:
                    self._n_classes = np.max(y) + 1
                self._n_features = X.shape[1]
                self._weight_maps, self._bias_maps = self._layermapping(
                    n_features=self._n_features,
                    n_classes=self._n_classes,
                    hidden_layers=self.hidden_layers)
                tf_weights, tf_biases = self._initialize_weights(
                    weight_maps=self._weight_maps,
                    bias_maps=self._bias_maps)
                self.cost_ = []
                self.train_acc_ = []
                self.valid_acc_ = []
            else:
                tf_weights, tf_biases = self._reuse_weights(
                    weights=self.weights_,
                    biases=self.biases_)

            # Prepare the training data
            y_enc = self._one_hot(y, self._n_classes)
            n_idx = list(range(y.shape[0]))
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            tf_y = tf.convert_to_tensor(value=y_enc, dtype=self.dtype)

            if validation:
                tf_X_valid = tf.convert_to_tensor(value=X_valid,
                                                  dtype=self.dtype)
                y_valid_enc = self._one_hot(y_valid, self._n_classes)
                tf_y_valid = tf.convert_to_tensor(value=y_valid_enc,
                                                  dtype=self.dtype)

            tf_idx = tf.placeholder(tf.int32,
                                    shape=[int(y.shape[0] / n_batches)])
            X_batch = tf.gather(params=tf_X, indices=tf_idx)
            y_batch = tf.gather(params=tf_y, indices=tf_idx)

            # Setup the graph for minimizing cross entropy cost
            net = self._predict(tf_X=tf_X,
                                tf_weights=tf_weights,
                                tf_biases=tf_biases,
                                activations=self.activations)

            # Define loss and optimizer
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net,
                                                                    tf_y)
            cost = tf.reduce_mean(cross_entropy)
            train = self.optimizer.minimize(cost)

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

                # compute prediction accuracy
                train_acc = self._accuracy(y, tf_X, tf_weights, tf_biases,
                                           self.activations)
                self.train_acc_.append(train_acc)
                if validation:
                    valid_acc = self._accuracy(y_valid, tf_X_valid,
                                               tf_weights, tf_biases,
                                               self.activations)
                    self.valid_acc_.append(valid_acc)
                else:
                    valid_acc = None
                self._print_progress(epoch + 1,
                                     cost=avg_cost,
                                     train_acc=train_acc,
                                     valid_acc=valid_acc)

            self.weights_ = {k: tf_weights[k].eval() for k in tf_weights}
            self.biases_ = {k: tf_biases[k].eval() for k in tf_biases}

        return

    def _accuracy(self, y, tf_X, tf_weights_, tf_biases_, activations):
        net = self._predict(tf_X=tf_X,
                            tf_weights=tf_weights_,
                            tf_biases=tf_biases_,
                            activations=activations)
        logits = tf.nn.softmax(net)
        y_pred = np.argmax(logits.eval(), axis=1)
        acc = np.sum(y == y_pred, axis=0) / float(y.shape[0])
        return acc

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
        self._check_arrays(X)
        if not hasattr(self, 'weights_'):
            raise AttributeError('The model has not been fitted, yet.')

        with tf.Session():
            tf.initialize_all_variables().run()
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            net = self._predict(tf_X=tf_X,
                                tf_weights=self.weights_,
                                tf_biases=self.biases_,
                                activations=self.activations)
            logits = tf.nn.softmax(net)
            return logits.eval()

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
        weights = {1: [[n_features, hidden_layers[0]],
                       'n_features, n_hidden_1'],
                   'out': [[hidden_layers[-1], n_classes],
                           'n_hidden_%d, n_classes' % len(hidden_layers)]}
        biases = {1: [[hidden_layers[0]], 'n_hidden_1'],
                  'out': [[n_classes], 'n_classes']}

        if len(hidden_layers) > 1:
            for i, h in enumerate(hidden_layers[1:]):
                layer = i + 2
                weights[layer] = [[weights[layer - 1][0][1], h],
                                  'n_hidden_%d, n_hidden_%d' % (layer -
                                                                1, layer)]
                biases[layer] = [[h], 'n_hidden_%d' % layer]
        return weights, biases

    def _predict(self, tf_X, tf_weights, tf_biases, activations):
        hidden_1 = self.activations[1](tf.add(tf.matmul(tf_X,
                                                        tf_weights[1]),
                                              tf_biases[1]))
        prev_layer = hidden_1
        if len(tf_weights) > 2:
            for layer in range(2, len(tf_weights)):
                prev_layer = self.activations[layer](tf.add(tf.matmul(
                    prev_layer, tf_weights[layer]), tf_biases[layer]))
        net = tf.matmul(prev_layer, tf_weights['out']) + tf_biases['out']
        return net

    def _reuse_weights(self, weights, biases):
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
            tf_weights[k[0]] = tf.Variable(tf.random_normal(
                weight_maps[k[0]][0], seed=seed))
            tf_biases[k[1]] = tf.zeros(bias_maps[k[1]][0])
        return tf_weights, tf_biases
