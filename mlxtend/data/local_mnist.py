# Sebastian Raschka 2014-2026
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

    Raises
    --------
    ValueError
        If an IDX header is incomplete or invalid, image and label counts
        differ, or a payload does not match its declared size. Image
        dimensions are read from the header rather than assumed to be 28x28.

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

    """
    with open(labels_path, "rb") as lbpath:
        header = lbpath.read(8)
        if len(header) != 8:
            raise ValueError("Incomplete MNIST label header: expected 8 bytes.")
        magic, n = struct.unpack(">II", header)
        if magic != 2049:
            raise ValueError("Invalid MNIST label magic number: expected 2049.")
        labels = np.fromfile(lbpath, dtype=np.uint8)
        if labels.size != n:
            raise ValueError(
                f"MNIST label payload has {labels.size} records; header declares {n}."
            )
    with open(images_path, "rb") as imgpath:
        header = imgpath.read(16)
        if len(header) != 16:
            raise ValueError("Incomplete MNIST image header: expected 16 bytes.")
        magic, num, rows, cols = struct.unpack(">IIII", header)
        if magic != 2051:
            raise ValueError("Invalid MNIST image magic number: expected 2051.")
        if num != n:
            raise ValueError(f"MNIST image count {num} does not match label count {n}.")
        if rows == 0 or cols == 0:
            raise ValueError("MNIST image dimensions must be positive.")
        images = np.fromfile(imgpath, dtype=np.uint8)
        expected_size = num * rows * cols
        if images.size != expected_size:
            raise ValueError(
                f"MNIST image payload has {images.size} bytes; "
                f"header declares {expected_size}."
            )
        images = images.reshape(num, rows * cols)

    return images, labels
