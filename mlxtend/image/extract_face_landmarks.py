# Sebastian Raschka 2014-2020
# contributor: Vahid Mirjalili
# mlxtend Machine Learning Library Extensions
#
# A counter class for printing the progress of an iterator.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import dlib
import numpy as np
import os
import warnings
from .utils import check_exists, download_url, extract_file

predictor_path = '~/mlxtend_data/shape_predictor_68_face_landmarks.dat'
predictor_url = ("http://dlib.net/files/"
                 "shape_predictor_68_face_landmarks.dat.bz2")

if not check_exists(predictor_path):
    download_url(predictor_url, save_path='~/mlxtend_data/')
    extract_file('~/mlxtend_data/shape_predictor_68_face_landmarks.dat.bz2')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.expanduser(predictor_path))


def extract_face_landmarks(img, return_dtype=np.int32):
    """Function to extract face landmarks.

    Note that this function requires an installation of
    the Python version of the library "dlib": http://dlib.net

    Parameters
    ----------

    img : array, shape = [h, w, ?]
        Numpy array of a face image or
        imageio.core.util.Array. E.g.,
        img = imageio.core.util.Array(ary)

        Supported shapes are
        - 3D tensors with 1
        or more color channels, for example,
        RGB: [h, w, 3]
        - 2D tensors without color channel, for example,
        Grayscale: [h, w]

    return_dtype: the return data-type of the array,
        default: np.int32.

    Returns
    ----------
    landmarks : numpy.ndarray, shape = [68, 2]
       A numpy array, where each row contains a landmark/point x-y coordinates.
       Return None if no face is detected by Dlib.

    Examples
    ----------
    For usage examples, please see
    http://rasbt.github.io/mlxtend/user_guide/image/extract_face_landmarks/

    """
    faces = detector(img, 1)  # detecting faces
    if not faces:
        warnings.warn('No face detected.')
        return None
    shape = predictor(img, faces[0])

    landmarks = np.zeros(shape=(68, 2))
    for i in range(68):
        p = shape.part(i)
        landmarks[i, :] = np.array([p.x, p.y])

    return landmarks.astype(return_dtype)
