# mlxtend Machine Learning Library Extensions
#
# A function for plotting a PCA correlation circle
# File Author: Gabriel Azevedo Ferreira <az.fe.gabriel@gmail.com>

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlxtend.externals.adjust_text import adjust_text
from mlxtend.feature_extraction import PrincipalComponentAnalysis


def corr2_coeff(A, B):
    """
    Compute correlation coefficients and return as a np array
    """
    A, B = np.array(A), np.array(B)
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def create_correlation_table(A, B, names_cols_A, names_cols_B):
    """
    Compute correlation coefficients and return as a DataFrame.

    A and B: 2d array like.
        The columns represent the different variables and the rows are
        the samples of thos variables
    names_cols_A/B : name to be added to the final pandas table

    return: pandas DataFrame with the corelations.Columns and Indexes
    represent the different variables of A and B (respectvely)
    """
    # corrs = np.corrcoef(np.transpose(A), np.transpose(B)
    #                     )[len(names_cols_A):, :len(names_cols_A)]
    corrs = corr2_coeff(A.T, B.T).T

    df_corrs = pd.DataFrame(corrs, columns=names_cols_A, index=names_cols_B)
    return df_corrs


def plot_pca_correlation_graph(
    X,
    variables_names,
    dimensions=(1, 2),
    figure_axis_size=6,
    X_pca=None,
    explained_variance=None,
):
    """
    Compute the PCA for X and plots the Correlation graph

    Parameters
    ----------
    X : 2d array like.
        The columns represent the different variables and the rows are the
         samples of thos variables

    variables_names : array like
        Name of the columns (the variables) of X

    dimensions: tuple with two elements.
        dimensions to be plotted (x,y)

    figure_axis_size :
         size of the final frame. The figure created is a square with length
         and width equal to figure_axis_size.

    X_pca : np.ndarray, shape = [n_samples, n_components].
        Optional.
        `X_pca` is the matrix of the transformed components from X.
        If not provided, the function computes PCA automatically using
        mlxtend.feature_extraction.PrincipalComponentAnalysis
        Expected `n_componentes >= max(dimensions)`

    explained_variance : 1 dimension np.ndarray, length = n_components
        Optional.
        `explained_variance` are the eigenvalues from the diagonalized
        covariance matrix on the PCA transformatiopn.
        If not provided, the function computes PCA independently
        Expected `n_componentes == X.shape[1]`

    Returns
    ----------
        matplotlib_figure, correlation_matrix

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/plotting/plot_pca_correlation_graph/

    """

    X = np.array(X)
    X = X - X.mean(axis=0)
    n_comp = max(dimensions)

    if (X_pca is None) and (explained_variance is None):
        pca = PrincipalComponentAnalysis(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)
        explained_variance = pca.e_vals_

    elif (X_pca is not None) and (explained_variance is None):
        raise ValueError(
            "If `X_pca` is not None, the `explained variance`"
            " values should not be `None`."
        )

    elif (X_pca is None) and (explained_variance is not None):
        raise ValueError(
            "If `explained variance` is not None, the `X_pca`"
            " values should not be `None`."
        )

    elif (X_pca is not None) and (explained_variance is not None):
        if X_pca.shape[1] != len(explained_variance):
            raise ValueError(
                f"Number of principal components must "
                f"match the number "
                f"of eigenvalues. Got "
                f"{X_pca.shape[1]} "
                f"!= "
                f"{len(explained_variance)}"
            )

    if X_pca.shape[1] < n_comp:
        raise ValueError(
            f"Input array `X_pca` contains fewer principal"
            f" components than expected based on `dimensions`."
            f" Got {X_pca.shape[1]} components in X_pca, expected"
            f" at least `max(dimensions)={n_comp}`."
        )
    if len(explained_variance) < n_comp:
        raise ValueError(
            f"Input array `explained_variance` contains fewer"
            f" elements than expected. Got"
            f" {len(explained_variance)} elements, expected"
            f"`X.shape[1]={X.shape[1]}`."
        )

    corrs = create_correlation_table(
        X_pca, X, ["Dim " + str(i + 1) for i in range(n_comp)], variables_names
    )
    tot = sum(X.var(0)) * X.shape[0] / (X.shape[0] - 1)
    explained_var_ratio = [(i / tot) * 100 for i in explained_variance]

    # Plotting circle
    fig_res = plt.figure(figsize=(figure_axis_size, figure_axis_size))
    plt.Circle((0, 0), radius=1, color="k", fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color="k", fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)

    # Plotting arrows
    texts = []
    for name, row in corrs.iterrows():
        x = row["Dim " + str(dimensions[0])]
        y = row["Dim " + str(dimensions[1])]
        plt.arrow(0.0, 0.0, x, y, color="k", length_includes_head=True, head_width=0.05)

        plt.plot([0.0, x], [0.0, y], "k-")
        texts.append(plt.text(x, y, name, fontsize=2 * figure_axis_size))
    # Plotting vertical lines
    plt.plot([-1.1, 1.1], [0, 0], "k--")
    plt.plot([0, 0], [-1.1, 1.1], "k--")

    # Adjusting text
    adjust_text(texts)
    # Setting limits and title
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.title("Correlation Circle", fontsize=figure_axis_size * 3)

    plt.xlabel(
        "Dim "
        + str(dimensions[0])
        + " (%s%%)" % str(explained_var_ratio[dimensions[0] - 1])[:4],
        fontsize=figure_axis_size * 2,
    )
    plt.ylabel(
        "Dim "
        + str(dimensions[1])
        + " (%s%%)" % str(explained_var_ratio[dimensions[1] - 1])[:4],
        fontsize=figure_axis_size * 2,
    )
    return fig_res, corrs
