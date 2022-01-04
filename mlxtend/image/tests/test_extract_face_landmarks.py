# Sebastian Raschka 2014-2022
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.image import extract_face_landmarks
import imageio
import numpy as np
import os
import pytest


if 'TRAVIS' in os.environ or os.environ.get('TRAVIS') == 'true':
    TRAVIS = True
else:
    TRAVIS = False


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_defaults():
    img = imageio.imread('mlxtend/image/tests/data/lena-small.png')
    landmarks1 = extract_face_landmarks(img)
    landmarks2 = extract_face_landmarks(np.asarray(img))
    np.testing.assert_array_equal(landmarks1, landmarks2)

    assert landmarks2.shape == (68, 2)

    true_vals = np.array([[85, 111],
                          [84, 120],
                          [85, 131],
                          [87, 140],
                          [91, 148],
                          [97, 155],
                          [104, 161],
                          [112, 164],
                          [120, 165],
                          [125, 162]], dtype=np.int32)
    np.testing.assert_array_equal(landmarks2[:10], true_vals)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_jpg():
    img = imageio.imread('mlxtend/image/tests/data/lena-small.jpg')
    landmarks1 = extract_face_landmarks(img)
    landmarks2 = extract_face_landmarks(np.asarray(img))
    np.testing.assert_array_equal(landmarks1, landmarks2)

    assert landmarks2.shape == (68, 2)

    true_vals = np.array([[85, 110],
                          [85, 120],
                          [85, 130],
                          [87, 140],
                          [91, 148],
                          [97, 155],
                          [104, 160],
                          [112, 164],
                          [120, 165],
                          [125, 162]], dtype=np.int32)

    if os.name == 'nt':
        # on windows, imageio parses jpgs sometimes differently so pixel values
        # maybe slightly different
        assert np.sum(np.abs(landmarks2[:10] - true_vals) > 2) == 0
    else:
        assert np.sum(np.abs(landmarks2[:10] - true_vals) > 0) == 0
        np.testing.assert_array_equal(landmarks2[:10], true_vals)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_grayscale():
    img = imageio.imread('mlxtend/image/tests/data/lena-small.png')
    img = rgb2gray(img)
    assert img.ndim == 2
    landmarks1 = extract_face_landmarks(img)

    assert landmarks1.shape == (68, 2)

    true_vals = np.array([[86, 111],
                          [85, 120],
                          [85, 130],
                          [87, 140],
                          [91, 148],
                          [98, 155],
                          [105, 160],
                          [113, 164],
                          [120, 165],
                          [126, 162]], dtype=np.int32)
    np.testing.assert_array_equal(landmarks1[:10], true_vals)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_noface():
    img = imageio.core.util.Array((
      np.random.random((193, 341, 3))*255).astype(np.uint8))
    landmarks = extract_face_landmarks(img)
    assert landmarks is None
