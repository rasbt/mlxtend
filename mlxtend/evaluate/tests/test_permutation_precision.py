from itertools import combinations, product

import numpy as np
import pytest

from mlxtend.evaluate import permutation_test


def _exact_integer_reference(x, y, paired):
    observed = abs(sum(x) - sum(y))
    if paired:
        statistics = [
            abs(sum(sign * (a - b) for sign, a, b in zip(signs, x, y)))
            for signs in product((-1, 1), repeat=len(x))
        ]
    else:
        pooled = x + y
        statistics = [
            abs(2 * sum(pooled[i] for i in indices) - sum(pooled))
            for indices in combinations(range(len(pooled)), len(x))
        ]
    return sum(value >= observed for value in statistics) / len(statistics)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("scale", [1e-12, 1.0, 1e12])
def test_exact_permutation_preserves_statistic_scale(paired, scale):
    x, y = [1, 2, 3], [4, 5, 6]

    def statistic(a, b):
        return scale * abs(np.mean(a) - np.mean(b))

    actual = permutation_test(x, y, func=statistic, paired=paired)

    assert actual == _exact_integer_reference(x, y, paired)


@pytest.mark.parametrize("paired", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sampled_permutation_preserves_statistic_scale(paired, dtype):
    x, y = [1, 2, 3], [4, 5, 6]
    kwargs = dict(paired=paired, method="approximate", num_rounds=200, seed=17)
    expected = permutation_test(x, y, **kwargs)

    def statistic(a, b):
        return dtype(1e-12 * abs(np.mean(a) - np.mean(b)))

    actual = permutation_test(x, y, func=statistic, **kwargs)

    assert actual == expected


def test_roundoff_equivalent_statistics_still_count_as_ties():
    x, y = [0.1, 0.2, 0.3], [0.4, 0.5, 0.6]

    assert permutation_test(x, y) == 0.1


def test_nearby_distinct_statistics_are_not_ties():
    x, y = [0, 1], [2, 3]

    def statistic(a, b):
        return 1.0 + 1e-7 * abs(np.mean(a) - np.mean(b))

    assert permutation_test(x, y, func=statistic) == 1 / 3


def test_integer_statistics_keep_exact_comparisons():
    x, y = [0, 1], [2, 3]

    def statistic(a, b):
        return 2**60 + abs(sum(map(int, a)) - sum(map(int, b)))

    assert permutation_test(x, y, func=statistic) == 1 / 3
