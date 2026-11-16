import struct

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from mlxtend.data import loadlocal_mnist


def _write_pair(tmp_path, label_header, image_header, labels, pixels):
    labels_path = tmp_path / "labels.idx1-ubyte"
    images_path = tmp_path / "images.idx3-ubyte"
    labels_path.write_bytes(label_header + labels)
    images_path.write_bytes(image_header + pixels)
    return images_path, labels_path


@pytest.mark.parametrize(
    "count,rows,cols", [(2, 28, 28), (3, 2, 3), (5, 1, 4), (0, 28, 28), (0, 2, 3)]
)
@pytest.mark.parametrize("as_string", [False, True])
def test_idx_header_dimensions_and_payload_are_preserved(
    tmp_path, count, rows, cols, as_string
):
    expected_pixels = (np.arange(count * rows * cols) % 256).astype(np.uint8)
    expected_labels = np.arange(count, dtype=np.uint8)
    paths = _write_pair(
        tmp_path,
        struct.pack(">II", 2049, count),
        struct.pack(">IIII", 2051, count, rows, cols),
        expected_labels.tobytes(),
        expected_pixels.tobytes(),
    )
    if as_string:
        paths = tuple(str(path) for path in paths)
    images, labels = loadlocal_mnist(*paths)
    assert images.shape == (count, rows * cols)
    assert images.dtype == labels.dtype == np.uint8
    assert images.flags.c_contiguous
    assert_array_equal(images.ravel(), expected_pixels)
    assert_array_equal(labels, expected_labels)


@pytest.mark.parametrize(
    "which,length", [("labels", 0), ("labels", 7), ("images", 0), ("images", 15)]
)
def test_truncated_idx_header_is_rejected(tmp_path, which, length):
    paths = _write_pair(
        tmp_path,
        struct.pack(">II", 2049, 2),
        struct.pack(">IIII", 2051, 2, 28, 28),
        bytes([3, 7]),
        bytes(1568),
    )
    path = paths[1 if which == "labels" else 0]
    path.write_bytes(path.read_bytes()[:length])
    with pytest.raises(ValueError):
        loadlocal_mnist(*paths)


@pytest.mark.parametrize(
    "label_magic,image_magic", [(0, 2051), (2049, 0), (2051, 2049)]
)
def test_wrong_idx_magic_is_rejected(tmp_path, label_magic, image_magic):
    paths = _write_pair(
        tmp_path,
        struct.pack(">II", label_magic, 2),
        struct.pack(">IIII", image_magic, 2, 28, 28),
        bytes([3, 7]),
        bytes(1568),
    )
    with pytest.raises(ValueError):
        loadlocal_mnist(*paths)


@pytest.mark.parametrize("rows,cols", [(0, 28), (28, 0), (0, 0)])
def test_zero_image_dimensions_are_rejected(tmp_path, rows, cols):
    paths = _write_pair(
        tmp_path,
        struct.pack(">II", 2049, 2),
        struct.pack(">IIII", 2051, 2, rows, cols),
        bytes([3, 7]),
        bytes(1568),
    )
    with pytest.raises(ValueError):
        loadlocal_mnist(*paths)


@pytest.mark.parametrize(
    "label_count,image_count,actual_labels,actual_pixels",
    [
        (2, 2, 1, 784),  # matching truncation previously passed silently
        (2, 2, 3, 2352),  # matching extra records previously passed silently
        (2, 3, 2, 1568),  # disagreeing header counts, same-sized payloads
        (3, 2, 2, 1568),
        (2, 2, 1, 1568),
        (2, 2, 2, 1567),
        (2, 2, 2, 1569),
        (0, 0, 1, 784),
        (2**32 - 1, 2**32 - 1, 1, 784),
    ],
)
def test_record_counts_and_payload_sizes_are_consistent(
    tmp_path, label_count, image_count, actual_labels, actual_pixels
):
    paths = _write_pair(
        tmp_path,
        struct.pack(">II", 2049, label_count),
        struct.pack(">IIII", 2051, image_count, 28, 28),
        bytes(actual_labels),
        bytes(actual_pixels),
    )
    with pytest.raises(ValueError):
        loadlocal_mnist(*paths)
