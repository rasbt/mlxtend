# Sebastian Raschka 2014-2019
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from .iris import iris_data
from .wine import wine_data
from .autompg import autompg_data
from .mnist import mnist_data
from .local_mnist import loadlocal_mnist
from .boston_housing import boston_housing_data
from .three_blobs import three_blobs_data
from .multiplexer import make_multiplexer_dataset

__all__ = ["iris_data", "wine_data", "autompg_data",
           "loadlocal_mnist", "mnist_data",
           "boston_housing_data", "three_blobs_data",
           "make_multiplexer_dataset"]
