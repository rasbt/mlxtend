## generalize_names_duplcheck



*generalize_names_duplcheck(df, col_name)*

Generalizes names and removes duplicates.

Applies mlxtend.text.generalize_names to a DataFrame with 1 first name letter
    by default and uses more first name letters if duplicates are detected.

**Parameters**


- `df` : `pandas.DataFrame`

    DataFrame that contains a column where generalize_names should be applied.


- `col_name` : `str`

    Name of the DataFrame column where `generalize_names` function should be applied to.

**Returns**


- `df_new` : `str`

    New DataFrame object where generalize_names function has been applied without duplicates.