

[![Build Status](https://travis-ci.org/rasbt/mlxtend.svg?branch=master)](https://travis-ci.org/rasbt/mlxtend)
[![Code Health](https://landscape.io/github/rasbt/mlxtend/master/landscape.svg?style=flat)](https://landscape.io/github/rasbt/mlxtend/master)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](http://badge.fury.io/py/mlxtend)
[![Coverage Status](https://coveralls.io/repos/rasbt/mlxtend/badge.svg?branch=master&service=github)](https://coveralls.io/github/rasbt/mlxtend?branch=master)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.5](https://img.shields.io/badge/python-3.5-blue.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

![](./docs/sources/img/logo.png)



**A library consisting of useful tools and extensions for the day-to-day data science tasks.**

<br>

Sebastian Raschka 2014-2016

Current version: 0.3.0

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


<hr>
## Recent changes

- Sequential Feature Selection algorithms: [SFS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_forward_selection/), [SFFS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_floating_forward_selection/), and [SFBS](http://rasbt.github.io/mlxtend/docs/feature_selection/sequential_floating_backward_selection/)
- [Neural Network / Multilayer Perceptron classifier](http://rasbt.github.io/mlxtend/docs/classifier/neuralnet_mlp/)
- [Ordinary least square regression](http://rasbt.github.io/mlxtend/docs/regression/linear_regression/) using different solvers (gradient and stochastic gradient descent, and the closed form solution)

<hr>
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

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from mlxtend.data import iris_data
from mlxtend.evaluate import plot_decision_regions

# Initializing Classifiers
clf1 = LogisticRegression(random_state=0)
clf2 = RandomForestClassifier(random_state=0)
clf3 = SVC(random_state=0, probability=True)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
X = X[:,[0, 2]]

# Plotting Decision Regions
gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()
```

![](./docs/sources/img/ensemble_decision_regions_2d.png)
