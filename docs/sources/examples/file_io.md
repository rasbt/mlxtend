<a id='file-io-utilities'></a>
## File IO Utilities

[[back to top](#overview)]

<br>

The `file_io utilities` can be imported via

	from mxtend.file_io import ...

<br>
<br>
<a id='find-files'></a>
### Find Files

[[back to top](#overview)]

##### Description

A function that finds files in a given directory based on substring matches and returns a list of the file names found.

##### Examples

	from mlxtend.file_io import find_files

    >>> find_files('mlxtend', '/Users/sebastian/Desktop')
	['/Users/sebastian/Desktop/mlxtend-0.1.6.tar.gz', 
	'/Users/sebastian/Desktop/mlxtend-0.1.7.tar.gz'] 
    

##### Default Parameters

    def find_files(substring, path, recursive=False, check_ext=None, ignore_invisible=True): 
        """
        Function that finds files in a directory based on substring matching.
        
        Parameters
        ----------
    
        substring : `str`
          Substring of the file to be matched.
    
        path : `str` 
          Path where to look.
    
        recursive: `bool`, optional, (default=`False`)
          If true, searches subdirectories recursively.
      
        check_ext: `str`, optional, (default=`None`)
          If string (e.g., '.txt'), only returns files that
          match the specified file extension.
      
        ignore_invisible : `bool`, optional, (default=`True`)
          If `True`, ignores invisible files (i.e., files starting with a period).
      
        Returns
        ----------
        results : `list`
          List of the matched files.
        
        """


<br>
<br>


<a id='find-file-groups'></a>
### Find File Groups

[[back to top](#overview)]

##### Description

A function that finds files that belong together (i.e., differ only by file extension) in different directories and collects them in a Python dictionary for further processing tasks. 

##### Examples

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/file_io_find_find_filegroups_1.png)

    d1 = os.path.join(master_path, 'dir_1')
    d2 = os.path.join(master_path, 'dir_2')
    d3 = os.path.join(master_path, 'dir_3')
    
    find_filegroups(paths=[d1,d2,d3], substring='file_1')
    # Returns:
    # {'file_1': ['/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_1/file_1.log', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_2/file_1.csv', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_3/file_1.txt']}
    #
    # Note: Setting `substring=''` would return a 
    # dictionary of all file paths for 
    # file_1.*, file_2.*, file_3.*

   
##### Default Parameters

    def find_filegroups(paths, substring='', extensions=None, validity_check=True, ignore_invisible=True):
        """
        Function that finds and groups files from different directories in a python dictionary.
        
        Parameters
        ----------
        paths : `list` 
          Paths of the directories to be searched. Dictionary keys are build from
          the first directory.
    
        substring : `str`, optional, (default=`''`)
          Substring that all files have to contain to be considered.
    
        extensions : `list`, optional, (default=`None`)
          `None` or `list` of allowed file extensions for each path. If provided, the number
          of extensions must match the number of `paths`.
         
        validity_check : `bool`, optional, (default=`True`)
          If `True`, checks if all dictionary values have the same number of file paths. Prints
          a warning and returns an empty dictionary if the validity check failed.

        ignore_invisible : `bool`, optional, (default=`True`)
          If `True`, ignores invisible files (i.e., files starting with a period).

        Returns
        ----------
        groups : `dict`
          Dictionary of files paths. Keys are the file names found in the first directory listed
          in `paths` (without file extension).
        
        """

<br>
<br>
<br>
<br>
