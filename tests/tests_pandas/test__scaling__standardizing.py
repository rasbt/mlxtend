from mlxtend.pandas import standardizing
import pandas as pd
import numpy as np

def test_minmax_scaling():
    s1 = pd.Series([1,2,3,4,5,6], index=(range(6)))
    s2 = pd.Series([10,9,8,7,6,5], index=(range(6)))
    df = pd.DataFrame(s1, columns=['s1'])
    df['s2'] = s2

    df_out1 = standardizing(df, ['s1','s2'])


    ary_out1 = np.array([[-1.33630621,  1.33630621],
       [-0.80178373,  0.80178373],
       [-0.26726124,  0.26726124],
       [ 0.26726124, -0.26726124],
       [ 0.80178373, -0.80178373],
       [ 1.33630621, -1.33630621]])

    np.testing.assert_allclose(df_out1.values, ary_out1, rtol=1e-03)