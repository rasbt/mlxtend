def _merge_lists(nested_list, high_level_indices=None):
    """
    merge elements of lists (of a nested_list) into one single tuple with elements
    sorted in ascending order.

    Parameters
    ----------
    nested_list: List
        a  list whose elements must be list as well.

    high_level_indices: list or tuple, default None
        a list or tuple that contains integers that are between 0 (inclusive) and
        the length of `nested_lst` (exclusive). If None, the merge of all
        lists nested in `nested_list` will be returned.

    Returns
    -------
    out: tuple
        a tuple, with elements sorted in ascending order, that is the merge of inner
        lists whose indices are provided in `high_level_indices`

    Example:
    nested_list = [[1],[2, 3],[4]]
    high_level_indices = [1, 2]
    >>> _merge_lists(nested_list, high_level_indices)
    (2, 3, 4) # merging [2, 3] and [4]
    """
    if high_level_indices is None:
        high_level_indices = list(range(len(nested_list)))

    out = []
    for idx in high_level_indices:
        out.extend(nested_list[idx])

    return tuple(sorted(out))
