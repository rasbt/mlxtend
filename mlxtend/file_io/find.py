import os

def find_files(substring, path, recursive=False): 
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
    return results