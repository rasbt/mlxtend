import os
import struct
import numpy as np


def loadlocal_mnist(images_path, labels_path):
    """ Read MNIST from ubyte files

    Parameters
    ----------
    images_path: str
        path to the test or train MNIST ubyte file
    labels_path: str
        path to the test or train MNIST class labels file

    Returns
    --------
    images: [n_samples, n_pixels] numpy.array
    labels: [n_samples] numpy array

    """
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII",
                                        imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels
