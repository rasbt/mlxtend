# Find Filegroups

A function that finds files in a given directory based on substring matches and returns a list of the file names found.

> from mlxtend.file_io import find_files

# Overview

This function finds files based on substring search. This is especially useful if we want to find specific files in a directory tree and return their absolute paths for further processing in Python.

### References

- -

### Related Topics

- [Find Files](./find_files.html)

# Examples

## Example 1 - Grouping related files in a dictionary

Given the following directory and file structure

    dir_1/
        file_1.log
        file_2.log
        file_3.log
    dir_2/
        file_1.csv
        file_2.csv
        file_3.csv
    dir_3/
        file_1.txt
        file_2.txt
        file_3.txt
        
we can use `find_files` to return the paths to all files that contain the substring `_2` as follows: 


```python
from mlxtend.file_io import find_files

find_files(substring='_2', path='./data_find_filegroups/', recursive=True)
```




    ['./data_find_filegroups/dir_1/file_2.log',
     './data_find_filegroups/dir_2/file_2.csv',
     './data_find_filegroups/dir_3/file_2.txt']



# API


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


