## plot_decision_regions

*plot_decision_regions(X, y, clf, X_highlight=None, res=0.02, legend=1, hide_spines=True, markers='s^oxv<>', colors=['red', 'blue', 'limegreen', 'gray', 'cyan'])*

Plot decision regions of a classifier.

**Parameters**

- `X` : array-like, shape = [n_samples, n_features]

    Feature Matrix.

- `y` : array-like, shape = [n_samples]

    True class labels.

- `clf` : Classifier object.

    Must have a .predict method.

- `X_highlight` : array-like, shape = [n_samples, n_features] (default: None)

    An array with data points that are used to highlight samples in `X`.

- `res` : float (default: 0.02)

    Grid width. Lower values increase the resolution but
    slow down the plotting.

- `hide_spines` : bool (default: True)

    Hide axis spines if True.

- `legend` : int (default: 1)

    Integer to specify the legend location.
    No legend if legend is 0.

- `markers` : list

    Scatterplot markers.

- `colors` : list

    Colors.

**Returns**

- `fig` : matplotlib.pyplot.figure object


