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
    assert len(find_files("data", tmp_path, recursive=True, ignore_invisible=False)) == 3
