# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.image import EyepadAlign
from mlxtend.image import extract_face_landmarks
import imageio
import numpy as np
import os


def test_defaults():
    path = 'mlxtend/image/tests/data/'
    eyepad = EyepadAlign()
    target_image = imageio.imread(os.path.join(path,
                                               'celeba-subset/000001.jpg'))
    eyepad.fit(target_image)

    img = imageio.imread(os.path.join(path, 'lena-small.png'))
    img_tr = eyepad.transform(img)

    landmarks_tr = extract_face_landmarks(img_tr)

    true_vals = np.array([[35, 113],
                          [33, 126],
                          [34, 140],
                          [36, 154],
                          [41, 166],
                          [51, 176],
                          [61, 184],
                          [72, 189],
                          [82, 190],
                          [90, 186]], dtype=np.int32)

    np.testing.assert_array_equal(landmarks_tr[:10], true_vals)


def test_fit2dir():
    path = 'mlxtend/image/tests/data/'
    eyepad = EyepadAlign()
    eyepad.fit(target_img_dir=os.path.join(path, 'celeba-subset/'),
               file_extensions='.jpg')

    img = imageio.imread(os.path.join(path, 'lena-small.png'))

    img_tr = eyepad.transform(img)

    landmarks_tr = extract_face_landmarks(img_tr)

    if os.name == 'nt':
        true_vals = np.array([[17, 66],
                              [17, 75],
                              [17, 85],
                              [19, 94],
                              [22, 102],
                              [28, 109],
                              [34, 114],
                              [42, 118],
                              [49, 118],
                              [54, 116]], dtype=np.int32)
    else:
        true_vals = np.array([[18, 68],
                              [18, 77],
                              [18, 86],
                              [19, 95],
                              [23, 103],
                              [29, 109],
                              [35, 114],
                              [43, 117],
                              [50, 118],
                              [55, 115]], dtype=np.int32)

    np.testing.assert_array_equal(landmarks_tr[:10], true_vals)
