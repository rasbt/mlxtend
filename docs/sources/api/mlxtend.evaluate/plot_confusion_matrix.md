## plot_confusion_matrix



*plot_confusion_matrix(conf_mat, hide_spines=False, hide_ticks=False, figsize=(2.5, 2.5), cmap=None, alpha=0.3)*

Plot a confusion matrix via matplotlib.

**Parameters**


- `conf_mat` : array-like, shape = [n_classes, n_classes]

    Confusion matrix from evaluate.confusion matrix.

- `hide_spines` : bool (default: False)

    Hides axis spines if True.

- `hide_ticks` : bool (default: False)

    Hides axis ticks if True

- `figsize` : tuple (default: (2.5, 2.5))

    Height and width of the figure

- `cmap` : matplotlib colormap (default: `None`)

    Uses matplotlib.pyplot.cm.Blues if `None`

**Returns**


- `fig, ax` : matplotlib.pyplot subplot objects

    Figure and axis elements of the subplot.