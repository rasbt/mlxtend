from mlxtend.file_io import find_filegroups, find_files


def test_file_group_suffix_is_literal(tmp_path):
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    for base in (left, right):
        (base / "value.txt").touch()
        (base / "cash$.txt").touch()
        (base / "price[1].txt").touch()
    paths = [left, right]
    assert set(find_filegroups(paths)) == {"value", "cash$", "price[1]"}
    assert set(find_filegroups(paths, rstrip="ue")) == {"val", "cash$", "price[1]"}
    assert set(find_filegroups(paths, rstrip="[1]")) == {"value", "cash$", "price"}


def test_recursive_hidden_paths(tmp_path):
    (tmp_path / ".private").mkdir()
    (tmp_path / ".private" / "data.txt").touch()
    (tmp_path / ".data.txt").touch()
    (tmp_path / "data.txt").touch()
    assert find_files("data", tmp_path, recursive=True) == [str(tmp_path / "data.txt")]
    assert (
        len(find_files("data", tmp_path, recursive=True, ignore_invisible=False)) == 3
    )


def test_unrestricted_secondary_extension(tmp_path):
    left, right = tmp_path / "left", tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "pair.txt").touch()
    (right / "pair.bin").touch()
    assert find_filegroups([left, right], extensions=[".txt", None]) == {
        "pair": [str(left / "pair.txt"), str(right / "pair.bin")]
    }


def test_invalid_group_arguments(tmp_path):
    import pytest

    with pytest.raises(ValueError, match="two paths"):
        find_filegroups([tmp_path])
    with pytest.raises(ValueError, match="one extension"):
        find_filegroups([tmp_path, tmp_path], extensions=[".txt"])
