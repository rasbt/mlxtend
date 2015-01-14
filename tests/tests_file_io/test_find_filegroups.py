from mlxtend.file_io import find_filegroups

import os

def test_find_files():
    master_path = os.path.join(os.path.abspath(os.getcwd()),'tests','data','find_filegroups')

    d1 = os.path.join(master_path, 'dir_1')
    d2 = os.path.join(master_path, 'dir_2')
    d3 = os.path.join(master_path, 'dir_3')
    
    
    ################################################
    # Example 1
    ################################################
    # {'file_1': ['/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_1/file_1.log', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_2/file_1.csv', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_3/file_1.txt']}
    
    
    f1_1 = os.path.join(master_path, d1, 'file_1.log')
    f1_2 = os.path.join(master_path, d2, 'file_1.csv')
    f1_3 = os.path.join(master_path, d3, 'file_1.txt')
    
    expect_dict_1 = {'file_1': [f1_1, f1_2, f1_3]}

    assert(find_filegroups(paths=[d1,d2,d3], substring='file_1') == expect_dict_1)
    
    
    ################################################
    # Example 2
    ################################################
    # {'file_3': ['/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_1/file_3.log', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_2/file_3.csv'], 
    # 'file_2': [' ...
    # }
    
    f2_1 = os.path.join(master_path, d1, 'file_1.log')
    f2_2 = os.path.join(master_path, d1, 'file_2.log')
    f2_3 = os.path.join(master_path, d1, 'file_3.log')
    f2_4 = os.path.join(master_path, d2, 'file_1.csv')
    f2_5 = os.path.join(master_path, d2, 'file_2.csv')
    f2_6 = os.path.join(master_path, d2, 'file_3.csv')

    expect_dict_2 = {'file_1': [f2_1, f2_4], 'file_2': [f2_2, f2_5], 'file_3': [f2_3, f2_6]}
    assert(find_filegroups(paths=[d1,d2], substring='') == expect_dict_2)
    
    
    
    ################################################
    # Example 3
    ################################################
    # {'file_1': ['/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_1/file_1.log']}
    
    f3_1 = os.path.join(master_path, d1, 'file_1.log')
    
    expect_dict_3 = {'file_1': [f3_1]}
    assert(find_filegroups(paths=[d1,d2], substring='file_1', extensions=['.log', '.abc']) == expect_dict_3)