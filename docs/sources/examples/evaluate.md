# Evaluate
<hr>
# Plotting Decision Regions

- A function to plot decision regions of classifiers.  

- Import `plot_decision_regions` via

    `from mlxtend.evaluate import plot_decision_regions`


<hr>

### 2D example


	from mlxtend.evaluate import plot_decision_regions
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.svm import SVC

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data[:, [0,2]]
	y = iris.target

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X,y)

	# Plotting decision regions
	plot_decision_regions(X, y, clf=svm, res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('SVM on Iris')
	plt.show()
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_2d.png)

<hr>

### 1D example



	from mlxtend.evaluate import plot_decision_regions
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.svm import SVC

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data[:, 2]
	X = X[:, None]
	y = iris.target

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X,y)

	# Plotting decision regions
	plot_decision_regions(X, y, clf=svm, res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('SVM on Iris')
	plt.show()
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_1d.png)


<hr>

### Highlighting Test Data Points


Via the `X_highlight`, a second dataset can be provided to highlight particular points in the dataset via a circle.

	from sklearn.cross_validation import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X_train, y_train)

	# Plotting decision regions

	plot_decision_regions(X, y, clf=svm, 
                      X_highlight=X_test, 
                      res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.title('SVM on Iris')
	plt.show()

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_highlight.png)

For more examples, please see this [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/evaluate_decision_regions.ipynb).


<hr>
<hr>

# Plotting Learning Curves
A function to plot learning curves for classifiers. Learning curves are extremely useful to analyze if a model is suffering from over- or under-fitting (high variance or high bias). The function can be imported via

    from mlxtend.evaluate import plot_learning_curves



<hr>
### Example



	from mlxtend.evaluate import plot_learning_curves
	from sklearn import datasets
	from sklearn.cross_validation import train_test_split

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_seed=2)

	from sklearn.tree import DecisionTreeClassifier
	import numpy as np

	clf = DecisionTreeClassifier(max_depth=1)

	plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='training_size')
	plt.show()

	plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='n_features')
	plt.show()


![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_learning_curves_1.png)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_learning_curves_2.png)

For more examples, please see this [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/evaluate_plot_learning_curves.ipynb)

