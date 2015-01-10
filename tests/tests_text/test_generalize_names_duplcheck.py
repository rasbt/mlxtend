from mlxtend.text import generalize_names_duplcheck
from mlxtend.text import generalize_names

import pandas as pd
import os

def test_generalize_names_duplcheck():
    
    path = os.path.join(os.getcwd(),'tests','data', 'csv', 'DKSalaries.csv')
    df = pd.read_csv(path)

    # duplicates before
    dupl = any(df['Name'].apply(generalize_names).duplicated())
    assert(dupl==True)
    
    # no duplicates
    df_new = generalize_names_duplcheck(df=df, col_name='Name')
    no_dupl = any(df_new['Name'].duplicated())
    assert(no_dupl==False)
   