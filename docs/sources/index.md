
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


## Example

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

![](./img/ensemble_decision_regions_2d.png)
