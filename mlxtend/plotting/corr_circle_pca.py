# mlxtend Machine Learning Library Extensions
#
# A function for removing chart junk from matplotlib plots
# File Author: Gabriel Azevedo Ferreira <az.fe.gabriel@gmail.com>

from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text


def create_correlation_table(A, B, names_cols_A, names_cols_B):
    """
    Compute the correlation coefficients and return as a DataFrame.

    A and B: 2d array like.
        The columns represent the different variables and the rows are the samples of thos variables
    names_cols_A/B : name to be added to the final pandas table

    return: pandas DataFrame with the corelations.Columns and Indexes represent the different variables of A and B (respectvely)
    """
    correlations = np.corrcoef(np.transpose(A),
                               np.transpose(B))[len(names_cols_A):, :len(names_cols_A)]
    df_correlations = pd.DataFrame(correlations,
                                   columns=names_cols_A, index=names_cols_B)
    return df_correlations


def plot_pca_correlation_graph(X, variables_names, dimensions=(1, 2), figure_axis_size=6, X_pca=None):
    """
    Compute the PCA for X and plots the Correlation graph

    X : 2d array like.
        The columns represent the different variables and the rows are the samples of thos variables
    names_cols_X : array like
        name to be added to the final pandas table.
    variables_names : array like
        Name of the columns (the variables) of X
    dimensions: tuple with two elements.
        dimensions to be plot (x,y)
    X_pca : optional. if not provided, compute PCA independently
    figure_axis_size :
         size of the final frame. The figure created is a square with length and width equal to figure_axis_size.
    """
    n_comp = max(dimensions)

    pca = PCA(n_components=n_comp)
    if X_pca is None:
        X_pca = pca.fit_transform(X)

    correlations = create_correlation_table(
        X_pca, X, ['Dim ' + str(i + 1) for i in range(n_comp)], variables_names)
    explained_var_ratio = pca.explained_variance_ratio_

    # Plotting circle
    fig_res = plt.figure(figsize=(figure_axis_size, figure_axis_size))
    plt.Circle((0, 0), radius=1, color='k', fill=False)
    circle1 = plt.Circle((0, 0), radius=1, color='k', fill=False)
    fig = plt.gcf()
    fig.gca().add_artist(circle1)

    # Plotting arrows
    texts = []
    for name, row in correlations.iterrows():
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
               str(explained_var_ratio[dimensions[0] - 1])[:4].lstrip("0."), fontsize=figure_axis_size * 2)
    plt.ylabel("Dim " + str(dimensions[1]) + " (%s%%)" %
               str(explained_var_ratio[dimensions[1] - 1])[:4].lstrip("0."), fontsize=figure_axis_size * 2)
    return fig_res, correlations
