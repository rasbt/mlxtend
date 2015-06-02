import os
import re

def find_files(substring, path, recursive=False, check_ext=None, ignore_invisible=True, ignore_substring=None):
    """
    Function that finds files in a directory based on substring matching.

    Parameters
    ----------

    substring : `str`
      Substring of the file to be matched.

    path : `str`
      Path where to look.

    recursive: `bool`
      If true, searches subdirectories recursively.

    check_ext: `str`
      If string (e.g., '.txt'), only returns files that
        match the specified file extension.

    ignore_invisible : `bool`
      If `True`, ignores invisible files (i.e., files starting with a period).

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




def find_filegroups(paths, substring='', extensions=None, validity_check=True, ignore_invisible=True, rstrip='', ignore_substring=None):
    """
    Function that finds and groups files from different directories in a python dictionary.

    Parameters
    ----------

    paths : `list`
      Paths of the directories to be searched. Dictionary keys are build from
        the first directory.

    substring : `str`
      Substring that all files have to contain to be considered.

    extensions : `list`
      `None` or `list` of allowed file extensions for each path. If provided, the number
        of extensions must match the number of `paths`.

    validity_check : `bool`
      If `True`, checks if all dictionary values have the same number of file paths. Prints
        a warning and returns an empty dictionary if the validity check failed.

    ignore_invisible : `bool`
      If `True`, ignores invisible files (i.e., files starting with a period).

    rstrip : `str` (default: "")
      If provided, strips characters from right side of the file base names after splitting
      the extension. Useful to trim different filenames to a common stem. E.g,.
      "abc_d.txt" and "abc_d_.csv" would share the stem "abc_d" if rstrip is set to "_".



    ignore_substring : `str`
      Ignores files that contain the specified substring.

    Returns
    ----------
    groups : `dict`
      Dictionary of files paths. Keys are the file names found in the first directory listed
        in `paths` (without file extension).

    """
    n = len(paths)

    # must have same number of paths and extensions
    assert(len(paths) >= 2)
    if extensions:
        assert(len(extensions) == n)
    else:
        extensions = ['' for i in range(n)]


    base = find_files(path=paths[0],  substring=substring, check_ext=extensions[0],
                      ignore_invisible=ignore_invisible, ignore_substring=ignore_substring)
    rest = [find_files(path=paths[i],  substring=substring, check_ext=extensions[i],
            ignore_invisible=ignore_invisible, ignore_substring=ignore_substring) for i in range(1,n)]

    groups = {}
    for f in base:
        basename = os.path.splitext(os.path.basename(f))[0]
        basename = re.sub('\%s$' % rstrip, '', basename)
        groups[basename] = [f]

    #groups = {os.path.splitext(os.path.basename(f))[0].rstrip(rstrip):[f] for f in base}

    for idx,r in enumerate(rest):
        for f in r:
            basename, ext = os.path.splitext(os.path.basename(f))
            basename = re.sub('\%s$' % rstrip, '', basename)
            try:
                if extensions[idx+1] == '' or ext == extensions[idx+1]:
                    groups[basename].append(f)
            except KeyError:
                pass

    if validity_check:
        lens = [len(groups[k]) for k in groups.keys()]
        if len(set(lens)) > 1:
            raise ValueError('Warning, some keys have more/less values than others. Set validity_check=False to ignore this warning.')

    return groups