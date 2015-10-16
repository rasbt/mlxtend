
![](./img/logo.png)

**A library consisting of useful tools and extensions for the day-to-day data science tasks.**

![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](http://badge.fury.io/py/mlxtend)

Sebastian Raschka 2014-2015

Current PyPI version: 0.2.9
Current GitHub version: 0.3.0dev


<br>

<hr>

## Links
- Documentation: [http://rasbt.github.io/mlxtend/](http://rasbt.github.io/mlxtend/)
- Source code repository: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)
- PyPI: [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend)
- Questions? Check out the [Google Groups mailing list](https://groups.google.com/forum/#!forum/mlxtend)

<br>

<hr>
## Recent changes

- Sequential Feature Selection algorithms: [SFS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_forward_selection/), [SFFS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_floating_forward_selection/), and [SFBS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_floating_backward_selection/)
- [Neural Network / Multilayer Perceptron classifier](http://rasbt.github.io/mlxtend/docs/classifier/neuralnet_mlp/)
- [Ordinary least square regression](http://rasbt.github.io/mlxtend/docs/regression/linear_regression/) using different solvers (gradient and stochastic gradient descent, and the closed form solution)

<hr>
<br>



## Quick Install

- latest version (from GitHub): `pip install git+git://github.com/rasbt/mlxtend.git#egg=mlxtend`
- latest PyPI version: `pip install mlxtend`

<hr>
<br>


## Examples

```python
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
```

![](./docs/evaluate/img/evaluate_plot_decision_regions_2d.png)
