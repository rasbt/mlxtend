# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# matplotlib utilities for removing chartchunk

from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def plot_decision_regions(X, y, clf, res=0.02, cycle_marker=True, legend=1):
    """
    Plots decision regions of a classifier.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features]
      Feature Matrix.

    y : array-like, shape = [n_samples]
      True class labels.

    clf : Classifier object. Must have a .predict method.

    res : float (default: 0.02)
      Grid width. Lower values increase the resolution but
        slow down the plotting.

    cycle_marker : bool
      Use different marker for each class.

    legend : int
      Integer to specify the legend location.
      No legend if legend is 0.

    cmap : Custom colormap object.
      Uses matplotlib.cm.rainbow if None.

    Returns
    ---------
    None

    Examples
    --------

    from sklearn import datasets
    from sklearn.svm import SVC

    iris = datasets.load_iris()
    X = iris.data[:, [0,2]]
    y = iris.target

    svm = SVC(C=1.0, kernel='linear')
    svm.fit(X,y)

    plot_decision_region(X, y, clf=svm, res=0.02, cycle_marker=True, legend=1)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('SVM on Iris')
    plt.show()

    """
    marker_gen = cycle('sxo^v')

    # make color map
    colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
    classes = np.unique(y)
    n_classes = len(np.unique(y))
    if n_classes > 5:
        raise NotImplementedError('Does not support more than 5 classes.')
    cmap = matplotlib.colors.ListedColormap(colors[:n_classes])

    # plot the decision surface

    # 2d
    if len(X.shape) == 2 and X.shape[1] > 1:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 1D
    else:
        y_min, y_max = -1, 1

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))

    # 2d
    if len(X.shape) == 2 and X.shape[1] > 1:
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        Z = clf.predict(np.array([xx.ravel(), yy.ravel()]).T)
    # 1D
    else:
        y_min, y_max = -1, 1
        Z = clf.predict(np.array([xx.ravel()]).T)

    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # plot class samples
    for c in np.unique(y):

        if len(X.shape) == 2 and X.shape[1] > 1:
            dim = X[y==c, 1]
        else:
            dim = [0 for i in X[y==c]]

        plt.scatter(X[y==c, 0],
                    dim,
                    alpha=0.8,
                    c=cmap(c),
                    marker=next(marker_gen),
                    label=c)

    if legend:
        plt.legend(loc=legend, fancybox=True, framealpha=0.5)