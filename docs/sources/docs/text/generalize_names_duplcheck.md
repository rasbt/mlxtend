mlxtend  
Sebastian Raschka, 05/14/2015


<hr>
# Name Generalization and Duplicates

**Note** that using [`generalize_names`](#name-generalization) with few `firstname_output_letters` can result in duplicate entries. E.g., if your dataset contains the names "Adam Johnson" and "Andrew Johnson", the default setting (i.e., 1 first name letter) will produce the generalized name "johnson a" in both cases.

One solution is to increase the number of first name letters in the output by setting the parameter `firstname_output_letters` to a value larger than 1. 

An alternative solution is to use the `generalize_names_duplcheck` function if you are working with pandas DataFrames. 

The  `generalize_names_duplcheck` function can be imported via

	from mlxtend.text import generalize_names_duplcheck

By default,  `generalize_names_duplcheck` will apply  `generalize_names` to a pandas DataFrame column with the minimum number of first name letters and append as many first name letters as necessary until no duplicates are present in the given DataFrame column. An example dataset column that contains the names  

<hr>
## Examples

Reading in a CSV file that has column `Name` for which we want to generalize the names:

- Samuel Eto'o
- Adam Johnson
- Andrew Johnson

<br>

    df = pd.read_csv(path)


Applying `generalize_names_duplcheck` to generate a new DataFrame with the generalized names without duplicates:	      

    df_new = generalize_names_duplcheck(df=df, col_name='Name')
- etoo s
- johnson ad
- johnson an

<hr>
## Default Parameters

<pre>def generalize_names_duplcheck(df, col_name):
    """
    Applies mlxtend.text.generalize_names to a DataFrame with 1 first name letter
    by default and uses more first name letters if duplicates are detected.

    Parameters
    ----------
    df : `pandas.DataFrame`
      DataFrame that contains a column where generalize_names should be applied.

    col_name : `str`
      Name of the DataFrame column where `generalize_names` function should be applied to.

    Returns
    ----------
    df_new : `str`
      New DataFrame object where generalize_names function has been applied without duplicates.

    """</pre>

