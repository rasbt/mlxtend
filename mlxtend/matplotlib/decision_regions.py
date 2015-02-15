# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions
# matplotlib utilities for removing chartchunk

import matplotlib.pyplot as plt
import numpy as np

def plot_decision_regions(X, y, clf, res=0.02):
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

    plot_decision_region(X, y, clf=svm, res=0.0001)

    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.title('SVM on Iris')
    plt.show()
    
    """
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res),
                         np.arange(y_min, y_max, res))
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y,  alpha=0.8)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())