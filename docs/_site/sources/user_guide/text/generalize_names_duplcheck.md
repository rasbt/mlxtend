# Generalize Names & Duplicate Checking

A function that converts a name into a general format ` <last_name><separator><firstname letter(s)> (all lowercase)` in a `pandas DataFrame` while avoiding duplicate entries.

> from mlxtend.text import generalize_names_duplcheck

## Overview

**Note** that using [`mlxtend.text.generalize_names`](./generalize_names.md) with few `firstname_output_letters` can result in duplicate entries. E.g., if your dataset contains the names "Adam Johnson" and "Andrew Johnson", the default setting (i.e., 1 first name letter) will produce the generalized name "johnson a" in both cases.

One solution is to increase the number of first name letters in the output by setting the parameter `firstname_output_letters` to a value larger than 1. 

An alternative solution is to use the `generalize_names_duplcheck` function if you are working with pandas DataFrames. 


By default,  `generalize_names_duplcheck` will apply  `generalize_names` to a pandas DataFrame column with the minimum number of first name letters and append as many first name letters as necessary until no duplicates are present in the given DataFrame column. An example dataset column that contains the names  

### References

- -

## Example 1 - Defaults

Reading in a CSV file that has column `Name` for which we want to generalize the names:

- Samuel Eto'o
- Adam Johnson
- Andrew Johnson


```python
import pandas as pd
from io import StringIO

simulated_csv = "name,some_value\n"\
                "Samuel Eto'o,1\n"\
                "Adam Johnson,1\n"\
                "Andrew Johnson,1\n"

df = pd.read_csv(StringIO(simulated_csv))
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>some_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Samuel Eto'o</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Adam Johnson</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Andrew Johnson</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



Applying `generalize_names_duplcheck` to generate a new DataFrame with the generalized names without duplicates: 


```python
from mlxtend.text import generalize_names_duplcheck
df_new = generalize_names_duplcheck(df=df, col_name='name')
df_new
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>some_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>etoo s</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>johnson ad</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>johnson an</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## API


*generalize_names_duplcheck(df, col_name)*

Generalizes names and removes duplicates.

Applies mlxtend.text.generalize_names to a DataFrame
with 1 first name letter by default
and uses more first name letters if duplicates are detected.

**Parameters**

- `df` : `pandas.DataFrame`

    DataFrame that contains a column where
    generalize_names should be applied.

- `col_name` : `str`

    Name of the DataFrame column where `generalize_names`
    function should be applied to.

**Returns**

- `df_new` : `str`

    New DataFrame object where generalize_names function has
    been applied without duplicates.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/text/generalize_names_duplcheck/](http://rasbt.github.io/mlxtend/user_guide/text/generalize_names_duplcheck/)


