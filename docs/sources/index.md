
<img src="./img/logo.png" alt="mlxtend logo" width="300px">

### Welcome to mlxtend's documentation!

**Mlxtend (machine learning extensions) is a Python library of useful tools for the day-to-day data science tasks.**

[![DOI](https://joss.theoj.org/papers/10.21105/joss.00638/status.svg)](https://doi.org/10.21105/joss.00638)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](https://badge.fury.io/py/mlxtend)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/mlxtend/badges/version.svg)](https://anaconda.org/conda-forge/mlxtend)
![Python 3](https://img.shields.io/badge/python-3-blue.svg)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](./license)
[![Discuss](https://img.shields.io/badge/discuss-github-blue.svg)](https://github.com/rasbt/mlxtend/discussions)

<hr>

## Links

- **Documentation:** [https://rasbt.github.io/mlxtend](https://rasbt.github.io/mlxtend)
- Source code repository: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)
- PyPI: [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend)
- Questions? Check out the [GitHub Discussions board](https://github.com/rasbt/mlxtend/discussions)

<hr>


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
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3],
                              weights=[2, 1, 1], voting='soft')

# Loading some example data
X, y = iris_data()
X = X[:,[0, 2]]

# Plotting Decision Regions

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(figsize=(10, 8))

labels = ['Logistic Regression',
          'Random Forest',
          'RBF kernel SVM',
          'Ensemble']

for clf, lab, grd in zip([clf1, clf2, clf3, eclf],
                         labels,
                         itertools.product([0, 1],
                         repeat=2)):
    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y,
                                clf=clf, legend=2)
    plt.title(lab)

plt.show()
```

---

![](./img/ensemble_decision_regions_2d.png)

If you use mlxtend as part of your workflow in a scientific publication, please consider citing the mlxtend repository with the following DOI:

[![DOI](https://joss.theoj.org/papers/10.21105/joss.00638/status.svg)](https://doi.org/10.21105/joss.00638)

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
  url          = {https://joss.theoj.org/papers/10.21105/joss.00638}
}
```


## License

- This project is released under a permissive new BSD open source license ([LICENSE-BSD3.txt](https://github.com/rasbt/mlxtend/blob/master/LICENSE-BSD3.txt)) and commercially usable. There is no warranty; not even for merchantability or fitness for a particular purpose.
- In addition, you may use, copy, modify and redistribute all artistic creative works (figures and images) included in this distribution under the directory
according to the terms and conditions of the Creative Commons Attribution 4.0 International License.  See the file [LICENSE-CC-BY.txt](https://github.com/rasbt/mlxtend/blob/master/LICENSE-CC-BY.txt) for details. (Computer-generated graphics such as the plots produced by matplotlib fall under the BSD license mentioned above).

## Contact

I received a lot of feedback and questions about mlxtend recently, and I thought that it would be worthwhile to set up a public communication channel. Before you write an email with a question about mlxtend, please consider posting it here since it can also be useful to others! Please join the [Google Groups Mailing List](https://groups.google.com/forum/#!forum/mlxtend)!

If Google Groups is not for you, please feel free to write me an [email](mailto:mail@sebastianraschka.com) or consider filing an issue on [GitHub's issue tracker](https://github.com/rasbt/mlxtend/issues) for new feature requests or bug reports. In addition, I setup a [Gitter channel](https://gitter.im/rasbt/mlxtend?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) for live discussions.
