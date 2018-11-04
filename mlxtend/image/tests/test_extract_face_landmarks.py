# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.image import extract_face_landmarks
import imageio
import numpy as np


def test_defaults():
    img = imageio.imread('mlxtend/image/tests/data/lena.png')
    landmarks = extract_face_landmarks(img)
    assert landmarks.shape == (68, 2)

    true_vals = np.array([[312, 376],
                          [285, 247],
                          [316, 319],
                          [274, 274],
                          [298, 345],
                          [272, 352]], dtype=np.int32)
    assert np.all(landmarks[np.arange(10, 70, 10)] == true_vals)
