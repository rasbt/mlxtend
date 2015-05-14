# Text Utilities

The `text utilities` can be imported via

	from mxtend.text import ...

<hr>
# Name Generalization

A function that converts a name into a general format ` <last_name><separator><firstname letter(s)> (all lowercase)`, which is useful if data is collected from different sources and is supposed to be compared or merged based on name identifiers. E.g., if names are stored in a pandas `DataFrame` column, the apply function can be used to generalize names: `df['name'] = df['name'].apply(generalize_names)`

<hr>
### Examples

	from mlxtend.text import generalize_names

    # defaults
    >>> generalize_names('Pozo, José Ángel')
    'pozo j'
    >>> generalize_names('Pozo, José Ángel') 
    'pozo j'
    >>> assert(generalize_names('José Ángel Pozo') 
    'pozo j' 
    >>> generalize_names('José Pozo')
    'pozo j' 
    
    # optional parameters
    >>> generalize_names("Eto'o, Samuel", firstname_output_letters=2)
    'etoo sa'
    >>> generalize_names("Eto'o, Samuel", firstname_output_letters=0)
    'etoo'
    >>> generalize_names("Eto'o, Samuel", output_sep=', ')
    'etoo, s' 

<hr>
# Name Generalization and Duplicates

**Note** that using [`generalize_names`](#name-generalization) with few `firstname_output_letters` can result in duplicate entries. E.g., if your dataset contains the names "Adam Johnson" and "Andrew Johnson", the default setting (i.e., 1 first name letter) will produce the generalized name "johnson a" in both cases.

One solution is to increase the number of first name letters in the output by setting the parameter `firstname_output_letters` to a value larger than 1. 

An alternative solution is to use the `generalize_names_duplcheck` function if you are working with pandas DataFrames. 

The  `generalize_names_duplcheck` function can be imported via

	from mlxtend.text import generalize_names_duplcheck

By default,  `generalize_names_duplcheck` will apply  `generalize_names` to a pandas DataFrame column with the minimum number of first name letters and append as many first name letters as necessary until no duplicates are present in the given DataFrame column. An example dataset column that contains the names  

<hr>
###Examples

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



