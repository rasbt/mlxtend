import numpy as np
import pytest

from mlxtend.evaluate import permutation_test


def count_greater(x, y):
    return np.count_nonzero(np.asarray(x) > np.asarray(y))


@pytest.mark.parametrize("container", [list, np.asarray])
@pytest.mark.parametrize("offset", [0, 2**53, 2**62])
def test_exact_paired_integer_statistics_are_translation_invariant(container, offset):
    x = container([offset + 1, offset + 1])
    y = container([offset, offset])
    result = permutation_test(x, y, paired=True, func=count_greater)

    # Of the four pairwise swaps, only the unchanged assignment has two wins.
    assert result == 0.25


@pytest.mark.parametrize("offset", [2**53, 2**62])
def test_sampled_paired_integer_statistics_are_translation_invariant(offset):
    x = np.array([1, 1, 1], dtype=np.int64)
    y = np.zeros(3, dtype=np.int64)
    options = dict(
        paired=True, func=count_greater, method="approximate", seed=17, num_rounds=50
    )
    reference = permutation_test(x, y, **options)
    actual = permutation_test(x + offset, y + offset, **options)

    assert actual == reference


def test_paired_samples_use_common_input_dtype():
    x = np.array([1, 2], dtype=np.int64)
    y = np.array([0.5, 1.5], dtype=np.float64)
    result = permutation_test(x, y, paired=True, func=count_greater)
    assert result == 0.25
    np.testing.assert_array_equal(x, [1, 2])
    np.testing.assert_array_equal(y, [0.5, 1.5])
