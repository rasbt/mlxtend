# Sebastian Raschka 2014-2024
# mlxtend Machine Learning Library Extensions
#
# A function for fetching the open-source MNIST dataset.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import struct

import numpy as np


def loadlocal_mnist(images_path, labels_path):
    """Read MNIST from ubyte files.

    Parameters
    ----------
    images_path : str
        path to the test or train MNIST ubyte file
    labels_path : str
        path to the test or train MNIST class labels file

    Returns
    --------
    images : [n_samples, n_pixels] numpy.array
        Pixel values of the images.
    labels : [n_samples] numpy array
        Target class labels

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

    """
    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
