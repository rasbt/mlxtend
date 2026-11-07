# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# A function for collecting file-group names from local directories.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os

from . import find_files


def find_filegroups(
    paths,
    substring="",
    extensions=None,
    validity_check=True,
    ignore_invisible=True,
    rstrip="",
    ignore_substring=None,
):
    """Find and collect files from different directories in a python dictionary.

    Parameters
    ----------
    paths : `list`
        Paths of the directories to be searched. Dictionary keys are build from
        the first directory.
    substring : `str` (default: '')
        Substring that all files have to contain to be considered.
    extensions : `list` (default: None)
        `None` or `list` of allowed file extensions for each path.
        Use None for an individual path to match all extensions.
        If provided, the number of extensions must match the number of `paths`.
    validity_check : `bool` (default: True)
        If `True`, checks if all dictionary values
        have the same number of file paths. Raises ValueError for unequal groups.
    ignore_invisible : `bool` (default: True)
        If `True`, ignores invisible files
        (i.e., files starting with a period).
    rstrip : `str` (default: '')
        If provided, removes this literal suffix once from each file
        base name after splitting the extension.
        Useful to trim different filenames to a common stem.
        E.g,. "abc_d.txt" and "abc_d_.csv" would share
        the stem "abc_d" if rstrip is set to "_".
    ignore_substring : `str` (default: None)
        Ignores files that contain the specified substring.

    Returns
    ----------
    groups : `dict`
        Dictionary of files paths. Keys are the file names
        found in the first directory listed
        in `paths` (without file extension).

    Examples
    -----------
    For usage examples, please see
    https://rasbt.github.io/mlxtend/user_guide/file_io/find_filegroups/

    """
    n = len(paths)

    # must have same number of paths and extensions
    if n < 2:
        raise ValueError("At least two paths are required")
    if extensions:
        if len(extensions) != n:
            raise ValueError("Expected one extension per path")
    else:
        extensions = ["" for i in range(n)]

    base = find_files(
        path=paths[0],
        substring=substring,
        check_ext=extensions[0],
        ignore_invisible=ignore_invisible,
        ignore_substring=ignore_substring,
    )
    rest = [
        find_files(
            path=paths[i],
            substring=substring,
            check_ext=extensions[i],
            ignore_invisible=ignore_invisible,
            ignore_substring=ignore_substring,
        )
        for i in range(1, n)
    ]

    groups = {}
    for f in base:
        basename = os.path.splitext(os.path.basename(f))[0]
        basename = basename.removesuffix(rstrip)
        groups[basename] = [f]

    for files in rest:
        for f in files:
            basename = os.path.splitext(os.path.basename(f))[0].removesuffix(rstrip)
            if basename in groups:
                groups[basename].append(f)

    if validity_check:
        lens = [len(groups[k]) for k in groups.keys()]
        if len(set(lens)) > 1:
            raise ValueError(
                "Warning, some keys have more/less values than"
                " others. Set validity_check=False"
                " to ignore this warning."
            )

    return groups

