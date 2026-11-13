# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os
import tempfile
import shutil

from mlxtend.file_io import find_filegroups, find_files


def _setup_dirs():
    tmp = tempfile.mkdtemp()
    for d in ("left", "right"):
        os.makedirs(os.path.join(tmp, d), exist_ok=True)
        open(os.path.join(tmp, d, "pair.txt"), "w").write("txt\n")
        open(os.path.join(tmp, d, "pair.bin"), "wb").write(b"\x00")
    return tmp


def test_find_filegroups_none_extension_accepts_all():
    # gh-1283: extensions=[None] must accept all extensions in the
    # secondary scan instead of dropping them.
    tmp = _setup_dirs()
    try:
        paths = [os.path.join(tmp, "left"), os.path.join(tmp, "right")]
        groups = find_filegroups(paths, extensions=[".txt", None])
        assert "pair" in groups
        assert len(groups["pair"]) == 3
        assert any(p.endswith("pair.bin") for p in groups["pair"])
    finally:
        shutil.rmtree(tmp)


def test_find_filegroups_preserves_literal_dollar_stem():
    # gh-1283: "$" in a filename is literal, not a regex anchor.
    tmp = _setup_dirs()
    try:
        open(os.path.join(tmp, "left", "cash$.txt"), "w").write("a\n")
        open(os.path.join(tmp, "right", "cash$.txt"), "w").write("b\n")
        paths = [os.path.join(tmp, "left"), os.path.join(tmp, "right")]
        # Intentionally unbalanced groups; validity of balance is not under test.
        groups = find_filegroups(paths, extensions=[None, None], validity_check=False)
        assert "cash$" in groups
        assert len(groups["cash$"]) == 2
    finally:
        shutil.rmtree(tmp)


def test_find_filegroups_rstrip_is_literal():
    # gh-1283: rstrip suffix is escaped, so multi-char suffixes strip
    # literal trailing characters rather than being regex-interpolated.
    tmp = _setup_dirs()
    try:
        open(os.path.join(tmp, "left", "abc_d.txt"), "w").write("v\n")
        open(os.path.join(tmp, "right", "abc_d_.csv"), "w").write("u\n")
        paths = [os.path.join(tmp, "left"), os.path.join(tmp, "right")]
        groups = find_filegroups(
            paths, extensions=[None, None], rstrip="_", validity_check=False
        )
        assert "abc_d" in groups
        assert len(groups["abc_d"]) == 2
    finally:
        shutil.rmtree(tmp)


def test_find_files_recursive_ignores_invisible():
    # gh-1283: recursive find_files honors ignore_invisible=True.
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, ".hidden_dir"), exist_ok=True)
        open(os.path.join(tmp, ".hidden_dir", "secret.txt"), "w").write("secret\n")
        open(os.path.join(tmp, "visible.txt"), "w").write("visible\n")
        results = find_files(
            substring="",
            path=tmp,
            recursive=True,
            check_ext=None,
            ignore_invisible=True,
            ignore_substring=None,
        )
        assert any(p.endswith("visible.txt") for p in results)
        assert not any(".hidden_dir" in p for p in results)
    finally:
        shutil.rmtree(tmp)
