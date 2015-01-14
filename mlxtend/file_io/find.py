import os

def find_files(substring, path, recursive=False, check_ext=None): 
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
      
    Returns
    ----------
    results : `list`
      List of the matched files.
        
    """
    def check_file(f, path):
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
            fn = check_file(f, path)
            if fn:
                results.append(fn)
                
    if check_ext:
        results = [r for r in results if os.path.splitext(r)[-1] == check_ext]
    
    return results
    
    
    

def find_filegroups(paths, substring='', extensions=None, validity_check=True):
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
    
    base = find_files(path=paths[0],  substring=substring, check_ext=extensions[0])
    rest = [find_files(path=paths[i],  substring=substring, check_ext=extensions[i]) for i in range(1,n)] 
    
    groups = {os.path.splitext(os.path.basename(f))[0]:[f] for f in base}
    
    for idx,r in enumerate(rest):
        for f in r:
            basename, ext = os.path.splitext(os.path.basename(f))
            try:
                if extensions[idx] == '' or ext == extensions[idx]:
                    groups[basename].append(f)
            except KeyError:
                pass
    
    if validity_check:
        num = sorted([(len(v),k) for k,v in groups.items()])
        for i in num[1:]:
            if i[0] < num[0][0]:
                print('Warning, key "%s" has less values than "key" %s.' % (i[1], num[0][1]))
                groups = {}

    return groups