# Sebastian Raschka 01/20/2015
# mlxtend Machine Learning Library Extensions
# pandas utilities for data scaling

def minmax_scaling(df, columns, min_val=0, max_val=1):
    """ 
    Min max scaling for pandas DataFrames
        
    Parameters
    ----------
    df : pandas DataFrame object.
      
    columns : array-like, shape = [n_columns]
      Array-like with pandas DataFrame column names, e.g., ['col1', 'col2', ...]
        
    min_val : `int` or `float`, optional (default=`0`)
      minimum value after rescaling.
    
    min_val : `int` or `float`, optional (default=`1`)
      maximum value after rescaling.
      
    Returns
    ----------
    
    df_new: pandas DataFrame object. 
      Copy of the DataFrame with rescaled columns.
      
    """
    df_new = df.copy()
    for c in columns:
        df_new[c] = (df_new[c] - df_new[c].min(axis=0)) / (df_new[c].max(axis=0) - df_new[c].min(axis=0))
        
        if not min_val == 0 and not max_val==1:
            df_new[c] = df_new[c] * (max_val - min_val) + min_val
        
    return df_new