# Sebastian Raschka 2014-2016
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import numpy as np
from nose.tools import raises
from mlxtend.evaluate import plot_decision_regions
from mlxtend.data import iris_data
from mlxtend.classifier import SoftmaxRegression
import matplotlib.pyplot as plt

plt.switch_backend('agg')
X, y = iris_data()
sr = SoftmaxRegression(epochs=15, eta=0.01, random_seed=1)


def test_pass():
    sr.fit(X[:, :2], y)
    plot_decision_regions(X=X[:, :2], y=y, clf=sr)


def test_ylist():
    sr.fit(X[:, :2], y)
    plot_decision_regions(X=X[:, :2], y=list(y), clf=sr)


@raises(ValueError)
def test_X_dim_fail():
    sr.fit(X, y)
    plot_decision_regions(X=X, y=y, clf=sr)


@raises(ValueError)
def test_y_dim_fail():
    sr.fit(X, y)
    plot_decision_regions(X=X, y=X, clf=sr)
