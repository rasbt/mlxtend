# mlxtend Machine Learning Library Extensions
#
# A function for plotting a PCA correlation circle
# File Author: Gabriel Azevedo Ferreira <az.fe.gabriel@gmail.com>

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mlxtend.externals.adjust_text import adjust_text
from mlxtend.feature_extraction import PrincipalComponentAnalysis


def create_correlation_table(A, B, names_cols_A, names_cols_B):
    """
    Compute the correlation coefficients and return as a DataFrame.

    A and B: 2d array like.
        The columns represent the different variables and the rows are
        the samples of thos variables
    names_cols_A/B : name to be added to the final pandas table

    return: pandas DataFrame with the corelations.Columns and Indexes
    represent the different variables of A and B (respectvely)
    """
    corrs = np.corrcoef(np.transpose(A), np.transpose(B)
                        )[len(names_cols_A):, :len(names_cols_A)]
    df_corrs = pd.DataFrame(corrs,
                            columns=names_cols_A, index=names_cols_B)
    return df_corrs


def plot_pca_correlation_graph(X, variables_names, dimensions=(1, 2),
                               figure_axis_size=6, X_pca=None):
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
        dimensions to be plot (x,y)
    X_pca : optional. if not provided, compute PCA independently
    figure_axis_size :
         size of the final frame. The figure created is a square with length
         and width equal to figure_axis_size.
    Returns
    ----------
        matplotlib_figure , correlation_matrix
    """
    X = np.array(X)
    X = X - X.mean(axis=0)
    n_comp = max(dimensions)

    if X_pca is None:
        pca = PrincipalComponentAnalysis(n_components=n_comp)
        pca.fit(X)
        X_pca = pca.transform(X)

    corrs = create_correlation_table(X_pca, X, ['Dim ' + str(i + 1) for i in
                                                range(n_comp)],
                                     variables_names)
    tot = sum(pca.e_vals_)
    explained_var_ratio = [(i / tot) * 100 for i in pca.e_vals_]

    # Plotting circle
    fig_res = plt.figure(figsize=(figure_axis_size, figure_axis_size))
    plt.Circle((0, 0), radius=1, color='k', fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color='k', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)

    # Plotting arrows
    texts = []
    for name, row in corrs.iterrows():
        x = row['Dim ' + str(dimensions[0])]
        y = row['Dim ' + str(dimensions[1])]
        plt.arrow(0.0, 0.0, x, y, color='k', length_includes_head=True,
                  head_width=.05)

        plt.plot([0.0, x], [0.0, y], 'k-')
        texts.append(plt.text(x, y, name, fontsize=2 * figure_axis_size))
    # Plotting vertical lines
    plt.plot([-1.1, 1.1], [0, 0], 'k--')
    plt.plot([0, 0], [-1.1, 1.1], 'k--')

    # Adjusting text
    adjust_text(texts)
    # Setting limits and title
    plt.xlim((-1.1, 1.1))
    plt.ylim((-1.1, 1.1))
    plt.title("Correlation Circle", fontsize=figure_axis_size * 3)

    plt.xlabel("Dim " + str(dimensions[0]) + " (%s%%)" %
               str(explained_var_ratio[dimensions[0] - 1])[:4].lstrip("0."),
               fontsize=figure_axis_size * 2)
    plt.ylabel("Dim " + str(dimensions[1]) + " (%s%%)" %
               str(explained_var_ratio[dimensions[1] - 1])[:4].lstrip("0."),
               fontsize=figure_axis_size * 2)
    return fig_res, corrs
