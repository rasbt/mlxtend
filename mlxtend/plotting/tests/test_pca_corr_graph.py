from mlxtend.plotting import plot_pca_correlation_graph
from mlxtend.data import iris_data
from sklearn.decomposition.pca import PCA

X, y = iris_data()


def test_pass_pca_corr():
    plot_pca_correlation_graph(X, ['1', '2', '3', '4'])


def test_pass_pca_corr_pca_out():
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    print(pca.singular_values_)
    print(pca.explained_variance_)
    print(pca.explained_variance_ratio_)

    plot_pca_correlation_graph(X, ['1', '2', '3', '4'], X_pca=X_pca, explained_variance=None)
