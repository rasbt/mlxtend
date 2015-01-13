from mlxtend.file_io import find_files

import os

def test_find_files():
    path = os.path.join(os.getcwd(),'tests','tests_file_io')
    expect = [os.path.join(os.getcwd(),'tests','tests_file_io', 'test_find_files.py')]

    assert(find_files(substring='test_find_files', path=path) == expect)
    
    # find recursive
    assert(find_files(substring='test_find_files.py', path=os.getcwd(), recursive=True) == expect)
    
    # find files and check extension
    assert(find_files(substring='test_find_files.py', path=path, check_ext='.py') == expect)
    assert(find_files(substring='test_find_files.py', path=path, check_ext='.txt') == [])
    