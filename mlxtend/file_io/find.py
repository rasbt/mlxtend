import os

def find_files(substring, path): 
    """
    Function that finds files in a directory based on substring matching.
        
    Parameters
    ----------
    substring : `str`
      Substring of the file to be matched.
    path : `str` 
      Path where to look.
      
    Returns
    ----------
    results : `list`
      List of the matched files.
        
    """
    results = []
    for f in os.listdir(path):
        if substring in f:
            compl_path = os.path.join(path, f)
            if os.path.isfile(compl_path):
                results.append(compl_path)
    return results