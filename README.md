[![DOI](https://joss.theoj.org/papers/10.21105/joss.00638/status.svg)](https://doi.org/10.21105/joss.00638)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](http://badge.fury.io/py/mlxtend)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mlxtend/badges/version.svg)](https://anaconda.org/conda-forge/mlxtend)
[![Build status](https://ci.appveyor.com/api/projects/status/7vx20e0h5dxcyla2/branch/master?svg=true)](https://ci.appveyor.com/project/rasbt/mlxtend/branch/master)
[![codecov](https://codecov.io/gh/rasbt/mlxtend/branch/master/graph/badge.svg)](https://codecov.io/gh/rasbt/mlxtend)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)
[![Discuss](https://img.shields.io/badge/discuss-github-blue.svg)](https://github.com/rasbt/mlxtend/discussions)

![](./docs/sources/img/logo.png)

**Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.**

<br>

Sebastian Raschka 2014-2022

<br>

## Links

- **Documentation:** [http://rasbt.github.io/mlxtend](http://rasbt.github.io/mlxtend)
- PyPI: [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend)
- Changelog: [http://rasbt.github.io/mlxtend/CHANGELOG](http://rasbt.github.io/mlxtend/CHANGELOG)
- Contributing: [http://rasbt.github.io/mlxtend/CONTRIBUTING](http://rasbt.github.io/mlxtend/CONTRIBUTING)
- Questions? Check out the [GitHub Discussions board](https://github.com/rasbt/mlxtend/discussions)

<br>
<br>

## Installing mlxtend

#### PyPI

To install mlxtend, just execute  

```bash
pip install mlxtend  
```

Alternatively, you could download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

```bash
python setup.py install
```

#### Conda
If you use conda, to install mlxtend just execute

```bash
conda install -c conda-forge mlxtend 
```

#### Dev Version

The mlxtend version on PyPI may always be one step behind; you can install the latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/rasbt/mlxtend.git#egg=mlxtend
```

Or, you can fork the GitHub repository from https://github.com/rasbt/mlxtend and install mlxtend from your local drive via

```bash
python setup.py install
```

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
from mlxtend.plotting import plot_decision_regions

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
                         ['Logistic Regression', 'Random Forest', 'RBF kernel SVM', 'Ensemble'],
                         itertools.product([0, 1], repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)
plt.show()
```

![](./docs/sources/img/ensemble_decision_regions_2d.png)

---

If you use mlxtend as part of your workflow in a scientific publication, please consider citing the mlxtend repository with the following DOI:


```
@article{raschkas_2018_mlxtend,
  author       = {Sebastian Raschka},
  title        = {MLxtend: Providing machine learning and data science 
                  utilities and extensions to Python’s  
                  scientific computing stack},
  journal      = {The Journal of Open Source Software},
  volume       = {3},
  number       = {24},
  month        = apr,
  year         = 2018,
  publisher    = {The Open Journal},
  doi          = {10.21105/joss.00638},
  url          = {http://joss.theoj.org/papers/10.21105/joss.00638}
}
```

- Raschka, Sebastian (2018) MLxtend: Providing machine learning and data science utilities and extensions to Python's scientific computing stack.
J Open Source Softw 3(24).

---

## License

- This project is released under a permissive new BSD open source license ([LICENSE-BSD3.txt](https://github.com/rasbt/mlxtend/blob/master/LICENSE-BSD3.txt)) and commercially usable. There is no warranty; not even for merchantability or fitness for a particular purpose.
- In addition, you may use, copy, modify and redistribute all artistic creative works (figures and images) included in this distribution under the directory
according to the terms and conditions of the Creative Commons Attribution 4.0 International License.  See the file [LICENSE-CC-BY.txt](https://github.com/rasbt/mlxtend/blob/master/LICENSE-CC-BY.txt) for details. (Computer-generated graphics such as the plots produced by matplotlib fall under the BSD license mentioned above).

## Contact

The best way to ask questions is via the [GitHub Discussions channel](https://github.com/rasbt/mlxtend/discussions). In case you encounter usage bugs, please don't hesitate to use the [GitHub's issue tracker](https://github.com/rasbt/mlxtend/issues) directly. 
