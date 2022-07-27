# Sebastian Raschka 2014-2022
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os

import imageio
import numpy as np
import pytest

from mlxtend.image import EyepadAlign, extract_face_landmarks

if "TRAVIS" in os.environ or os.environ.get("TRAVIS") == "true":
    TRAVIS = True
else:
    TRAVIS = False


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_defaults():
    path = "mlxtend/image/tests/data/"
    eyepad = EyepadAlign()
    target_image = imageio.imread(os.path.join(path, "celeba-subset/000001.jpg"))
    eyepad.fit_image(target_image)

    img = imageio.imread(os.path.join(path, "lena-small.png"))
    img_tr = eyepad.transform(img)

    landmarks_tr = extract_face_landmarks(img_tr)

    true_vals = np.array(
        [
            [35, 113],
            [33, 126],
            [34, 140],
            [36, 154],
            [41, 166],
            [51, 176],
            [61, 184],
            [72, 189],
            [82, 190],
            [90, 186],
        ],
        dtype=np.int32,
    )

    if os.name == "nt":
        # on windows, imageio parses jpgs sometimes differently so pixel values
        # maybe slightly different
        assert np.sum(np.abs(landmarks_tr[:10] - true_vals) > 2) == 0
    else:
        assert np.sum(np.abs(landmarks_tr[:10] - true_vals) > 0) == 0, np.sum(
            np.abs(landmarks_tr[:10] - true_vals)
        )
        np.testing.assert_array_equal(landmarks_tr[:10], true_vals)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_fit2dir():
    path = "mlxtend/image/tests/data/"
    eyepad = EyepadAlign()
    eyepad.fit_directory(
        target_img_dir=os.path.join(path, "celeba-subset/"),
        target_width=178,
        target_height=218,
        file_extension=".jpg",
    )

    img = imageio.imread(os.path.join(path, "lena-small.png"))

    img_tr = eyepad.transform(img)

    landmarks_tr = extract_face_landmarks(img_tr)

    true_vals = np.array(
        [
            [30, 115],
            [28, 130],
            [29, 145],
            [31, 160],
            [38, 173],
            [48, 182],
            [59, 191],
            [71, 196],
            [82, 197],
            [90, 193],
        ],
        dtype=np.int32,
    )

    if os.name == "nt":
        # on windows, imageio parses jpgs sometimes differently so pixel values
        # maybe slightly different
        assert np.sum(np.abs(landmarks_tr[:10] - true_vals) > 3) == 0
    else:
        assert np.sum(np.abs(landmarks_tr[:10] - true_vals) > 0) == 0, np.sum(
            np.abs(landmarks_tr[:10] - true_vals) > 0
        )
        np.testing.assert_array_equal(landmarks_tr[:10], true_vals)


@pytest.mark.skipif(TRAVIS, reason="DLIB download too slow")
def test_empty_dir():
    path = "mlxtend/image/tests/data/"
    eyepad = EyepadAlign()

    def tmp_func():
        eyepad.fit_directory(
            target_img_dir=os.path.join(path, "celeba-subset/"),
            target_width=178,
            target_height=218,
            file_extension=".PNG",
        )

    with pytest.raises(ValueError):
        tmp_func()
