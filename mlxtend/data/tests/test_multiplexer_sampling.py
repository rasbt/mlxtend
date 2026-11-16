import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from mlxtend.data import make_multiplexer_dataset

# Recorded from the scalar-draw implementation at upstream 1aee767e.
# Big-endian int64 bytes make the fixtures independent of host byte order.
with Path(__file__).with_name("multiplexer_seeded_outputs.json").open() as f:
    SEEDED_OUTPUTS = json.load(f)


@pytest.mark.parametrize("case", SEEDED_OUTPUTS)
def test_vectorized_draws_preserve_complete_seeded_datasets(case):
    X, y = make_multiplexer_dataset(**case["parameters"])
    digest = hashlib.sha256(X.astype(">i8").tobytes() + y.astype(">i8").tobytes())
    assert digest.hexdigest() == case["sha256"]
    assert X.dtype == np.dtype(np.int_)
    assert y.dtype == np.dtype(np.int_)

    address_bits = case["parameters"]["address_bits"]
    addresses = X[:, :address_bits] @ (2 ** np.arange(address_bits - 1, -1, -1))
    np.testing.assert_array_equal(y, X[np.arange(len(X)), address_bits + addresses])
    assert np.isin(X, [0, 1]).all()
    assert int(y.sum()) == case["positives"]
    if not case["parameters"]["shuffle"]:
        np.testing.assert_array_equal(
            y, [1] * case["positives"] + [0] * (len(y) - case["positives"])
        )


def test_local_seed_does_not_advance_global_numpy_generator():
    before = np.random.get_state()
    make_multiplexer_dataset(
        address_bits=5, sample_size=31, shuffle=True, random_seed=8
    )
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
