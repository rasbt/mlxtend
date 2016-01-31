# Find Filegroups

A function that finds files that belong together (i.e., differ only by file extension) in different directories and collects them in a Python dictionary for further processing tasks. 

> from mlxtend.file_io import find_filegroups

# Overview

This function finds files that are related to each other based on their file names. This can be useful for parsing collections files that have been stored in different subdirectories, for examples:

    input_dir/
        task01.txt
        task02.txt
        ...
    log_dir/
        task01.log
        task02.log
        ...
    output_dir/
        task01.dat
        task02.dat
        ...

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
        
we can use `find_filegroups` to group related files as items of a dictionary as shown below:


```python
from mlxtend.file_io import find_filegroups

find_filegroups(paths=['./data_find_filegroups/dir_1', 
                       './data_find_filegroups/dir_2', 
                       './data_find_filegroups/dir_3'], 
                substring='file_')
```




    {'file_1': ['./data_find_filegroups/dir_1/file_1.log',
      './data_find_filegroups/dir_2/file_1.csv',
      './data_find_filegroups/dir_3/file_1.txt'],
     'file_2': ['./data_find_filegroups/dir_1/file_2.log',
      './data_find_filegroups/dir_2/file_2.csv',
      './data_find_filegroups/dir_3/file_2.txt'],
     'file_3': ['./data_find_filegroups/dir_1/file_3.log',
      './data_find_filegroups/dir_2/file_3.csv',
      './data_find_filegroups/dir_3/file_3.txt']}



# API


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


