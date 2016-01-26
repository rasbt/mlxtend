## find_filegroups



*find_filegroups(paths, substring='', extensions=None, validity_check=True, ignore_invisible=True, rstrip='', ignore_substring=None)*

Find and collect files from different directories in a python dictionary.

**Parameters**


- `paths` : `list`

    Paths of the directories to be searched. Dictionary keys are build from
    the first directory.

- `substring` : `str` (default: '')

    Substring that all files have to contain to be considered.

- `extensions` : `list` (default: None)

    `None` or `list` of allowed file extensions for each path.
    If provided, the number of extensions must match the number of `paths`.

- `validity_check` : `bool` (default: None)

    If `True`, checks if all dictionary values
    have the same number of file paths. Prints
    a warning and returns an empty dictionary if the validity check failed.

- `ignore_invisible` : `bool` (default: True)

    If `True`, ignores invisible files
    (i.e., files starting with a period).

- `rstrip` : `str` (default: '')

    If provided, strips characters from right side of the file
    base names after splitting the extension.
    Useful to trim different filenames to a common stem.
    E.g,. "abc_d.txt" and "abc_d_.csv" would share
    the stem "abc_d" if rstrip is set to "_".

- `ignore_substring` : `str` (default: None)

    Ignores files that contain the specified substring.

**Returns**


- `groups` : `dict`

    Dictionary of files paths. Keys are the file names
    found in the first directory listed
    in `paths` (without file extension).