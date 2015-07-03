

[![Build Status](https://travis-ci.org/rasbt/mlxtend.svg?branch=dev)](https://travis-ci.org/rasbt/mlxtend)
[![Code Health](https://landscape.io/github/rasbt/mlxtend/master/landscape.svg?style=flat)](https://landscape.io/github/rasbt/mlxtend/master)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](http://badge.fury.io/py/mlxtend)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

![](./docs/sources/img/logo.png)



**A library consisting of useful tools and extensions for the day-to-day data science tasks.**

<br>

Sebastian Raschka 2014-2015

Current version: 0.2.8

<br>


## Links

- **Documentation:** [http://rasbt.github.io/mlxtend/](http://rasbt.github.io/mlxtend/)
- Source code repository: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)
- PyPI: [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend)
- Changelog: [http://rasbt.github.io/mlxtend/changelog](http://rasbt.github.io/mlxtend/changelog)
- Contributing: [http://rasbt.github.io/mlxtend/contributing](http://rasbt.github.io/mlxtend/contributing)
- Questions? Check out the [Google Groups mailing list](https://groups.google.com/forum/#!forum/mlxtend)

<br>
<br>

## Recent changes

- [Neural Network / Multilayer Perceptron classifier](http://rasbt.github.io/mlxtend/docs/classifier/neuralnet_mlp/)
- [5000 labeled training samples](http://rasbt.github.io/mlxtend/docs/data/mnist/) from the MNIST handwritten digits dataset
- [Ordinary least square regression](http://rasbt.github.io/mlxtend/docs/regression/linear_regression/) using different solvers (gradient and stochastic gradient descent, and the closed form solution)


<br>
<br>


## Installing mlxtend

To install `mlxtend`, just execute  

    pip install mlxtend  


The `mlxtend` version on PyPI may always one step behind; you can install the latest development version from this GitHub repository by executing

    pip install git+git://github.com/rasbt/mlxtend.git#egg=mlxtend

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

    python setup.py install 


<br>
<br>


## Examples

	from mlxtend.evaluate import plot_decision_regions
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.svm import SVC

	### Loading some example data
	iris = datasets.load_iris()
	X = iris.data[:, [0,2]]
	y = iris.target

	### Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X,y)

	### Plotting decision regions
	plot_decision_regions(X, y, clf=svm, res=0.02, legend=2)

	### Adding axes annotations
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('SVM on Iris')
	plt.show()

![](./docs/sources/img/evaluate_plot_decision_regions_2d.png)




