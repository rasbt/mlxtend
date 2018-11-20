# Sebastian Raschka 2014-2018
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

from mlxtend.image import extract_face_landmarks
import imageio
import numpy as np
import os


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)


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


def test_jpg():
    img = imageio.imread('mlxtend/image/tests/data/lena-small.jpg')
    landmarks1 = extract_face_landmarks(img)
    landmarks2 = extract_face_landmarks(np.asarray(img))
    np.testing.assert_array_equal(landmarks1, landmarks2)

    assert landmarks2.shape == (68, 2)

    expected_img = np.array([226, 136, 127, 224, 134, 125, 224, 134, 123,
                             227, 138, 124, 228, 136, 121, 224, 133, 115,
                             224, 131, 113, 225, 132, 114, 225, 132, 115,
                             223, 130, 113, 227, 137, 128, 225, 135, 124,
                             225, 136, 122, 229, 137, 122, 228, 137, 119,
                             224, 133, 114, 224, 131, 113, 225, 132, 114,
                             223, 130, 112, 222, 129, 111, 228, 136, 123,
                             226, 134, 121, 226, 134, 119, 227, 136, 118,
                             227, 134, 116, 224, 132, 111, 223, 131, 110,
                             224, 132, 111, 222, 130, 109, 224, 129, 111,
                             225, 133, 118, 225, 132, 117, 225, 132, 115,
                             225, 132, 114, 225, 133, 112, 223, 131, 108,
                             224, 129, 107, 224, 129, 107, 224, 129, 109,
                             223, 128, 108, 224, 131, 114, 225, 130, 112,
                             225, 130, 112, 226, 131, 111, 226, 131, 109,
                             225, 131, 106, 224, 130, 105, 225, 131, 106,
                             225, 130, 108, 224, 129, 107, 226, 131, 113,
                             225, 130, 110, 225, 130, 110, 227, 131, 109,
                             227, 131, 107, 227, 131, 107, 227, 131, 107,
                             228, 132, 108, 227, 131, 107, 224, 129, 107,
                             227, 130, 111, 227, 130, 111, 226, 130, 108,
                             226, 130, 106, 226, 130, 106, 227, 131, 106,
                             227, 131, 106, 227, 131, 107, 228, 132, 108,
                             224, 129, 107, 228, 129, 110, 228, 129, 110,
                             228, 129, 108, 227, 128, 105, 227, 128, 105,
                             228, 129, 106, 226, 130, 106, 226, 130, 106,
                             228, 132, 108, 223, 128, 106, 229, 130, 111,
                             228, 129, 110, 227, 128, 107, 227, 128, 107,
                             229, 130, 107, 230, 131, 108, 231, 132, 109,
                             229, 133, 109, 227, 131, 109, 224, 129, 107,
                             227, 128, 109, 227, 128, 109, 227, 128, 109,
                             228, 129, 108, 230, 131, 108, 231, 132, 109,
                             229, 133, 109, 228, 132, 110, 227, 131, 109,
                             224, 129, 109])

    np.testing.assert_array_equal((img[:10, :10]).flatten(), expected_img)

    if os.name == 'nt':
        true_vals = np.array([[85, 111],
                              [84, 120],
                              [85, 130],
                              [87, 140],
                              [91, 148],
                              [98, 155],
                              [105, 160],
                              [113, 164],
                              [120, 165],
                              [126, 162]], dtype=np.int32)

    else:
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

    np.testing.assert_array_equal(landmarks2[:10], true_vals)


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


def test_noface():
    img = imageio.core.util.Array((
      np.random.random((193, 341, 3))*255).astype(np.uint8))
    landmarks = extract_face_landmarks(img)
    assert landmarks.shape == (68, 2)
    np.testing.assert_array_equal(landmarks, np.zeros((68, 2)))
