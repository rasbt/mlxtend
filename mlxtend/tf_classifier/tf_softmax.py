# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Implementation of Softmax Regression in Tensorflow
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import tensorflow as tf
import numpy as np
from time import time
from .._base import _BaseModel
from .._base import _IterativeModel
from .._base import _MultiClass
from .._base import _Classifier


class TfSoftmaxRegression(_BaseModel, _IterativeModel, _MultiClass,
                          _Classifier):
    """Softmax regression classifier.

    Parameters
    ------------
    eta : float (default: 0.5)
        Learning rate (between 0.0 and 1.0)
    epochs : int (default: 50)
        Passes over the training dataset.
        Prior to each epoch, the dataset is shuffled
        if `minibatches > 1` to prevent cycles in stochastic gradient descent.
    n_classes : int (default: None)
        A positive integer to declare the number of class labels
        if not all class labels are present in a partial training set.
        Gets the number of class labels automatically if None.
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
    def __init__(self, eta=0.5, epochs=50, n_classes=None,
                 minibatches=1, random_seed=None,
                 print_progress=0, dtype=None):

        if dtype is None:
            self.dtype = tf.float32
        else:
            self.dtype = dtype
        self.eta = eta
        self.epochs = epochs
        self.n_classes = n_classes
        self.minibatches = minibatches
        self.random_seed = random_seed
        self.print_progress = print_progress
        self._is_fitted = False

    def _fit(self, X, y, init_params=True,):
        self._check_target_array(y)

        n_batches = self.minibatches
        if y.shape[0] % n_batches != 0:
            raise AttributeError("Training set size %d cannot"
                                 " be divided into %d minibatches without"
                                 " remainder" % (y.shape[0], n_batches))

        # Construct the Graph
        g = tf.Graph()
        with g.as_default():

            if init_params:
                if self.n_classes is None:
                    self.n_classes = np.max(y) + 1
                self._n_features = X.shape[1]
                tf_w_, tf_b_ = self._initialize_weights(
                    n_features=self._n_features,
                    n_classes=self.n_classes)
                self.cost_ = []
                self.train_acc_ = []
                self.valid_acc_ = []
            else:
                tf_w_ = tf.Variable(self.w_)
                tf_b_ = tf.Variable(self.b_)

            # Prepare the training data
            y_enc = self._one_hot(y, self.n_classes, dtype=np.float)
            tf_X = tf.convert_to_tensor(value=X, dtype=self.dtype)
            tf_y = tf.convert_to_tensor(value=y_enc, dtype=self.dtype)

            tf_idx = tf.placeholder(tf.int32,
                                    shape=[int(y.shape[0] / n_batches)])
            X_batch = tf.gather(params=tf_X, indices=tf_idx)
            y_batch = tf.gather(params=tf_y, indices=tf_idx)

            # Setup the graph for minimizing cross entropy cost
            net = tf.matmul(X_batch, tf_w_) + tf_b_
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(net,
                                                                    y_batch)
            cost = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=self.eta)
            train = optimizer.minimize(cost)

            # Initializing the variables
            init = tf.global_variables_initializer()

        # Launch the graph
        with tf.Session(graph=g) as sess:
            sess.run(init)
            rgen = np.random.RandomState(self.random_seed)
            self.init_time_ = time()
            for epoch in range(self.epochs):
                costs = []
                for idx in self._yield_minibatches_idx(
                        rgen=rgen,
                        n_batches=self.minibatches,
                        data_ary=y,
                        shuffle=True):

                    _, c = sess.run([train, cost], feed_dict={tf_idx: idx})
                    costs.append(c)
                avg_cost = np.mean(costs)
                self.cost_.append(avg_cost)

                # compute prediction accuracy
                train_acc = self._accuracy(y, tf_X, tf_w_, tf_b_)
                self.train_acc_.append(train_acc)
                if self.print_progress:
                    self._print_progress(iteration=epoch + 1,
                                         n_iter=self.epochs,
                                         cost=avg_cost)

            self.w_ = tf_w_.eval()
            self.b_ = tf_b_.eval()

    def _accuracy(self, y, tf_X, tf_w_, tf_b_):
        logits = tf.nn.softmax(tf.matmul(tf_X, tf_w_) +
                               tf_b_)
        y_pred = np.argmax(logits.eval(), axis=1)
        acc = np.sum(y == y_pred, axis=0) / float(y.shape[0])
        return acc

    def _resuse_weights(self, weights, biases):
            w = tf.Variable(weights)
            b = tf.Variable(biases)
            return w, b

    def _initialize_weights(self, n_features, n_classes):
            w = tf.Variable(tf.truncated_normal([n_features, n_classes],
                                                seed=self.random_seed))
            b = tf.Variable(tf.zeros([n_classes]))
            return w, b

    def _predict(self, X):
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
            logits = tf.nn.softmax(tf.matmul(tf_X, self.w_) +
                                   self.b_)
            return logits.eval()
