import matplotlib.pyplot as plt
import numpy as np


def plot_decision_regions_3d(
    X,
    y,
    clf,
    z_slices,
    feature_index=(0, 1, 2),
    ax=None,
    res=0.02,
    scatter_points=True,
    alpha=0.3,
):
    """
    Stack 2D decision regions in a 3D space.
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    markers = ("s", "x", "o", "^", "v")

    x_min, x_max = X[:, feature_index[0]].min() - 1, X[:, feature_index[0]].max() + 1
    y_min, y_max = X[:, feature_index[1]].min() - 1, X[:, feature_index[1]].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, res), np.arange(y_min, y_max, res))

    for z_val in z_slices:
        n_points = xx.ravel().shape[0]
        grid_points = np.c_[xx.ravel(), yy.ravel(), np.full(n_points, z_val)]

        Z = clf.predict(grid_points)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, zdir="z", offset=z_val, alpha=alpha, cmap="RdYlBu")
    if scatter_points:
        for idx, cl in enumerate(np.unique(y)):
            ax.scatter(
                X[y == cl, feature_index[0]],
                X[y == cl, feature_index[1]],
                X[y == cl, feature_index[2]],
                alpha=0.8,
                c=colors[idx % len(colors)],
                marker=markers[idx % len(markers)],
                label=f"Class {cl}",
            )
    ax.set_xlabel(f"Feature {feature_index[0]}")
    ax.set_ylabel(f"Feature {feature_index[1]}")
    ax.set_zlabel(f"Feature {feature_index[2]}")
    ax.legend(loc="upper left")

    return ax
