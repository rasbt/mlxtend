mlxtend  
Sebastian Raschka, last updated: 07/14/2015



<hr>
# Logistic Regression

> from mlxtend.classifier import LogisticRegression

Implementation of Logistic Regression  with different learning rules: Gradient descent and stochastic gradient descent.

![](./img/classifier_logistic_regression_schematic.png)

For more usage examples please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/classifier_logistic_regression.ipynb).

A more detailed article about the algorithms is in preparation.


<hr>
### Example 1 - Stochastic Gradient Descent


	from mlxtend.data import iris_data
	from mlxtend.evaluate import plot_decision_regions
	from mlxtend.classifier import LogisticRegression
	import matplotlib.pyplot as plt

	# Loading Data

	X, y = iris_data()
	X = X[:, [0, 3]] # sepal length and petal width
	X = X[0:100] # class 0 and class 1
	y = y[0:100] # class 0 and class 1

	# standardize
	X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()



	lr = LogisticRegression(eta=0.01, epochs=100, learning='sgd')
	lr.fit(X, y)

	plot_decision_regions(X, y, clf=lr)
	plt.title('Logistic Regression - Stochastic Gradient Descent')
	plt.show()

	print(lr.w_)

	plt.plot(range(len(lr.cost_)), lr.cost_)
	plt.xlabel('Iterations')
	plt.ylabel('Missclassifications')
	plt.show()



![](./img/classifier_logistic_regression_sgd_1.png)

![](./img/classifier_logistic_regression_sgd_2.png)



<hr>
### Default Parameters

<pre>class LogisticRegression(object):
    """Logistic regression classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)

    epochs : int
      Passes over the training dataset.

    learning : str (default: sgd)
      Learning rule, sgd (stochastic gradient descent)
      or gd (gradient descent).

    regularization : {None, 'l2'} (default: None)
      Type of regularization. No regularization if
      `regularization=None`.

    l2_lambda : float
      Regularization parameter for L2 regularization.
      No regularization if ls_lambda_=0.0.

    shuffle : bool (default: False)
        Shuffles training data every epoch if True to prevent circles.

    random_seed : int (default: None)
        Set random state for shuffling and initializing the weights.

    zero_init_weight : bool (default: False)
        If True, weights are initialized to zero instead of small random
        numbers in the interval [0,1]

    """</pre>


<hr>
### Methods

<pre>    def fit(self, X, y, init_weights=True):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        init_weights : bool (default: True)
            (Re)initializes weights to small random floats if True.

        Returns
        -------
        self : object

        """</pre>
        
        
        
<pre>    def predict(self, X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        class : int
          Predicted class label.

        """</pre>
        
        
        
        
<pre>    def activation(self, X):
        """
        Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
          Class 1 probability : float

        """
        
        </pre>
