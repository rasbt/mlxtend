# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Implementation of a Multi-layer Perceptron in Tensorflow
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import tensorflow as tf
import numpy as np
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _MultiClass
from .._base import _MultiLayer
from .._base import _Classifier


class TfMultiLayerPerceptron(_BaseModel, _IterativeModel,
                             _MultiClass, _MultiLayer, _Classifier):
    """Multi-layer perceptron classifier.

    Parameters
    ------------
    eta : float (default: 0.5)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    hidden_layers : list (default: [50, 10])
        Number of units per hidden layer. By default 50 units in the
        first hidden layer, and 10 hidden units in the second hidden layer.
    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.
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
    dropout : float (default: 1.0)
        A float between in the range (0.0, 1.0] to specify
        the probability that each element is kept.
    decay : list, shape=[decay_rate, decay_steps] (default: [0.0, 1])
        Parameter to specify the exponential decay of the learning rate eta
        for adaptive learning (eta * decay_rate ^ (epoch / decay_steps)).
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
    w_ : 2d-array, shape=[n_features, n_classes]
        Weights after fitting.
    b_ : 1D-array, shape=[n_classes]
        Bias units after fitting.
    cost_ : list
        List of floats, the average cross_entropy for each epoch.

    """
    def __init__(self, eta=0.5, epochs=50,
                 hidden_layers=[50, 10],
                 n_classes=None,
                 activations=['logistic', 'logistic'],
                 optimizer='gradientdescent',
                 momentum=0.0, l1=0.0, l2=0.0,
                 dropout=1.0,
                 decay=[0.0, 1.0],
                 minibatches=1, random_seed=None,
                 print_progress=0, dtype=None):

        self.eta = eta
        if len(hidden_layers) != len(activations):
            raise AttributeError('Number of hidden_layers and'
                                 ' n_activations must be equal.')
        self.hidden_layers = hidden_layers
        self.momentum = momentum
        self.n_classes = n_classes
        self.activations = self._get_activations(activations)
        self.l1 = l1
        self.l2 = l2
        self.dropout = dropout
        self.decay = decay
        self.optimizer = optimizer
        self._init_optimizer(self.optimizer)
        self.epochs = epochs
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

        if dtype is None:
            self.dtype = tf.float32
        else:
            self.dtype = dtype

    def _init_optimizer(self, optimizer):
        self.global_step_ = tf.Variable(0, trainable=False)
        if self.decay[0] > 0.0:
            learning_rate = tf.train.exponential_decay(
                learning_rate=self.eta,
                global_step=self.global_step_,
                decay_steps=self.decay[1],
                decay_rate=self.decay[0])

        else:
            learning_rate = self.eta
        if optimizer == 'gradientdescent':
            opt = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                             momentum=self.momentum)
        elif optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optimizer == 'ftrl':
            opt = tf.train.FtrlOptimizer(
                learning_rate=learning_rate,
                l1_regularization_strength=self.l1,
                l2_regularization_strength=self.l2)
        elif optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)
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
            act[str(idx + 1)] = adict[a]
        return act

    def _fit(self, X, y, init_params=True):
        self._check_target_array(y)
        n_batches = self.minibatches
        if y.shape[0] % n_batches != 0:
            raise AttributeError("Training set size %d cannot"
                                 " be divided into %d minibatches without"
                                 " remainder" % (y.shape[0], n_batches))

        # Construct the Graph
        g = tf.Graph()
        with g.as_default():
            self.optimizer_ = self._init_optimizer(self.optimizer)
            if init_params:
                if self.n_classes is None:
                    self.n_classes = np.max(y) + 1

                self._n_features = X.shape[1]
                self._weight_maps, self._bias_maps = self._layermapping(
                    n_features=self._n_features,
                    n_classes=self.n_classes,
                    hidden_layers=self.hidden_layers)
                tf_weights, tf_biases = self._init_params_from_layermapping(
                    weight_maps=self._weight_maps,
                    bias_maps=self._bias_maps,
                    activations=self.activations)
                self.cost_ = []

            else:
                tf_weights, tf_biases = self._reuse_weights(
                    weights=self.w_,
                    biases=self.b_)

            # Prepare the training data
            y_enc = self._one_hot(y, self.n_classes, dtype=np.float)
            n_idx = list(range(y.shape[0]))
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            tf_y = tf.convert_to_tensor(value=y_enc, dtype=self.dtype)

            tf_idx = tf.placeholder(tf.int32,
                                    shape=[int(y.shape[0] / n_batches)])
            X_batch = tf.gather(params=tf_X, indices=tf_idx)
            y_batch = tf.gather(params=tf_y, indices=tf_idx)

            # Setup the graph for minimizing cross entropy cost
            net = self._predict(tf_X=X_batch,
                                tf_weights=tf_weights,
                                tf_biases=tf_biases,
                                activations=self.activations,
                                dropout=True)

            # Define loss and optimizer
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net,
                                                                    y_batch)
            cost = tf.reduce_mean(cross_entropy)
            train = self.optimizer_.minimize(cost,
                                             global_step=self.global_step_)

            # Initializing the variables
            init = tf.global_variables_initializer()

        # random seed for shuffling and dropout
        rgen = np.random.RandomState(self.random_seed)

        # Launch the graph
        with tf.Session(graph=g) as sess:
            sess.run(init)
            self.init_time_ = time()
            for epoch in range(self.epochs):
                if self.minibatches > 1:
                    n_idx = rgen.permutation(n_idx)
                minis = np.array_split(n_idx, self.minibatches)
                costs = []
                for idx in minis:
                    _, c = sess.run([train, cost], feed_dict={tf_idx: idx})
                    costs.append(c)
                avg_cost = np.mean(costs)
                self.cost_.append(avg_cost)

                self._print_progress(iteration=epoch + 1,
                                     n_iter=self.epochs,
                                     cost=avg_cost)

            self.w_ = {k: tf_weights[k].eval() for k in tf_weights}
            self.b_ = {k: tf_biases[k].eval() for k in tf_biases}

    def _accuracy(self, y, tf_X, tf_w_, tf_biases_, activations):
        net = self._predict(tf_X=tf_X,
                            tf_weights=tf_w_,
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
        if not hasattr(self, 'w_'):
            raise AttributeError('The model has not been fitted, yet.')

        with tf.Session():
            tf.global_variables_initializer().run()
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            net = self._predict(tf_X=tf_X,
                                tf_weights=self.w_,
                                tf_biases=self.b_,
                                activations=self.activations,
                                dropout=False)
            logits = tf.nn.softmax(net)
            return logits.eval()

    def _predict(self, tf_X, tf_weights, tf_biases,
                 activations, dropout=False):
        hidden_1 = self.activations['1'](tf.add(tf.matmul(tf_X,
                                                          tf_weights['1']),
                                                tf_biases['1']))
        prev_layer = hidden_1
        if dropout:
            prev_layer = tf.nn.dropout(prev_layer, self.dropout,
                                       seed=self.random_seed)
        if len(tf_weights) > 2:
            for layer in range(2, len(tf_weights)):
                layer = str(layer)
                prev_layer = self.activations[layer](tf.add(tf.matmul(
                    prev_layer, tf_weights[layer]), tf_biases[layer]))
                if dropout:
                    prev_layer = tf.nn.dropout(prev_layer, dropout,
                                               seed=self.random_seed)
        net = tf.matmul(prev_layer, tf_weights['out']) + tf_biases['out']
        return net

    def _reuse_weights(self, weights, biases):
            w = {k: tf.Variable(self.w_[k]) for k in self.w_}
            b = {k: tf.Variable(self.b_[k]) for k in self.b_}
            return w, b

    def _init_params_from_layermapping(self, weight_maps,
                                       bias_maps, activations):
        tf_weights, tf_biases = {}, {}
        for i, k in enumerate(zip(sorted(weight_maps), sorted(bias_maps))):
            assert k[0] == k[1]
            if self.random_seed:
                seed = self.random_seed + i
            else:
                seed = None
            tf_weights[k[0]] = tf.Variable(tf.random_normal(
                weight_maps[k[0]][0], seed=seed))

            if k[0] in activations and activations[k[0]] != 'relu':
                tf_biases[k[1]] = tf.zeros(bias_maps[k[1]][0])
            else:
                tf_biases[k[0]] = tf.constant(0.1, shape=bias_maps[k[1]][0])
        return tf_weights, tf_biases
