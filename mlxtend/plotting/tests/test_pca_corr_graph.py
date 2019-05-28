from mlxtend.plotting import plot_pca_correlation_graph
from mlxtend.data import iris_data

X, y = iris_data()


def test_pass_pca_corr():
    plot_pca_correlation_graph(X, ['1', '2', '3', '4'])
