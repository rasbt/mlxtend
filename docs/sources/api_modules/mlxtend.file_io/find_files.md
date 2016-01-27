## find_files

*find_files(substring, path, recursive=False, check_ext=None, ignore_invisible=True, ignore_substring=None)*

Find files in a directory based on substring matching.

**Parameters**

- `substring` : `str`

    Substring of the file to be matched.

- `path` : `str`

    Path where to look.
    recursive: `bool`
    If true, searches subdirectories recursively.
    check_ext: `str`
    If string (e.g., '.txt'), only returns files that
    match the specified file extension.

- `ignore_invisible` : `bool`

    If `True`, ignores invisible files
    (i.e., files starting with a period).

- `ignore_substring` : `str`

    Ignores files that contain the specified substring.

**Returns**

- `results` : `list`

    List of the matched files.

