# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# A function for searching files in local directories.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import os


def find_files(substring, path, recursive=False,
               check_ext=None, ignore_invisible=True,
               ignore_substring=None):
    """Find files in a directory based on substring matching.

    Parameters
    ----------
    substring : `str`
        Substring of the file to be matched.
    path : `str`
        Path where to look.
    recursive : `bool`
        If true, searches subdirectories recursively.
    check_ext : `str`
        If string (e.g., '.txt'), only returns files that
        match the specified file extension.
    ignore_invisible : `bool`
        If `True`, ignores invisible files
        (i.e., files starting with a period).
    ignore_substring : `str`
        Ignores files that contain the specified substring.

    Returns
    ----------
    results : `list`
        List of the matched files.

    """
    def check_file(f, path):
        if not (ignore_substring and ignore_substring in f):
            if substring in f:
                compl_path = os.path.join(path, f)
                if os.path.isfile(compl_path):
                    return compl_path
        return False

    results = []

    if recursive:
        for par, nxt, fnames in os.walk(path):
            for f in fnames:
                fn = check_file(f, par)
                if fn:
                    results.append(fn)

    else:
        for f in os.listdir(path):
            if ignore_invisible and f.startswith('.'):
                continue
            fn = check_file(f, path)
            if fn:
                results.append(fn)

    if check_ext:
        results = [r for r in results if os.path.splitext(r)[-1] == check_ext]

    return results
