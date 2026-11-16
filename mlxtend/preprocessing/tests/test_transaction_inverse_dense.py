import numpy as np
import pytest

from mlxtend.preprocessing import TransactionEncoder


@pytest.mark.parametrize("seed", [0, 7, 41])
@pytest.mark.parametrize("n_items", [0, 1, 13, 2049])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_dense_decoding_roundtrip_preserves_transaction_and_item_order(
    seed, n_items, layout
):
    rng = np.random.RandomState(seed)
    vocabulary = ["item-%04d" % i for i in range(n_items)]
    transactions = [
        list(rng.choice(vocabulary, size=rng.randint(0, 8))) if n_items else []
        for _ in range(17)
    ]
    encoder = TransactionEncoder().fit([vocabulary[::-1]])
    data = encoder.transform(transactions)
    if layout == "F":
        data = np.asfortranarray(data)
    elif layout == "strided":
        storage = np.zeros((data.shape[0], data.shape[1] * 2), dtype=bool)
        storage[:, ::2] = data
        data = storage[:, ::2]
    original = data.copy()

    decoded = encoder.inverse_transform(data)

    assert decoded == [sorted(set(transaction)) for transaction in transactions]
    np.testing.assert_array_equal(data, original)


@pytest.mark.parametrize("n_items", [0, 7])
def test_no_transactions_returns_empty_list(n_items):
    encoder = TransactionEncoder().fit([list(range(n_items))])
    assert encoder.inverse_transform(np.zeros((0, n_items), dtype=bool)) == []


@pytest.mark.parametrize("dtype", [bool, np.int8, np.float64, object])
def test_dense_numeric_and_object_truth_values_preserve_item_selection(dtype):
    encoder = TransactionEncoder().fit([["z", "b", "a", "d"]])
    data = np.array([[0, -2, 0, 3], [1, 0, -1, 0]], dtype=dtype)
    assert encoder.inverse_transform(data) == [["b", "z"], ["a", "d"]]


def test_dense_python_rows_and_independent_result_lists():
    encoder = TransactionEncoder().fit([["a", "b", "c"]])
    data = [[True, False, True], [True, False, True], [False, False, False]]
    result = encoder.inverse_transform(data)
    assert result == [["a", "c"], ["a", "c"], []]
    result[0].append("extra")
    assert result[1] == ["a", "c"]
    assert data == [[True, False, True], [True, False, True], [False, False, False]]
