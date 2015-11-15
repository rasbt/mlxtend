
Sebastian Raschka, 2015  
`mlxtend`, a library of extension and helper modules for Python's data analysis and machine learning libraries

- GitHub repository: https://github.com/rasbt/mlxtend
- Documentation: http://rasbt.github.io/mlxtend/


```python
%load_ext watermark
%watermark -a 'Sebastian Raschka' -u -d -v -p matplotlib,numpy,scipy
```

    Sebastian Raschka 
    Last updated: 11/15/2015 
    
    CPython 3.5.0
    IPython 4.0.0
    
    matplotlib 1.5.0
    numpy 1.10.1
    scipy 0.16.0



```python
import sys
sys.path.insert(0, '../../../github_mlxtend/')

import mlxtend
mlxtend.__version__
```




    '0.3.0dev'



# Generalize Names

A function that converts a name into a general format ` <last_name><separator><firstname letter(s)> (all lowercase)`.

> from mlxtend.text import generalize_names

# Overview


A function that converts a name into a general format ` <last_name><separator><firstname letter(s)> (all lowercase)`, which is useful if data is collected from different sources and is supposed to be compared or merged based on name identifiers. E.g., if names are stored in a pandas `DataFrame` column, the apply function can be used to generalize names: `df['name'] = df['name'].apply(generalize_names)`

### References

- -

### Related Topics

- [Name Generalization and Duplicates](./generalize_names_duplcheck.html)
- [Tokenizer](./tokenizer.html)

# Examples

## Example 1 - Defaults


```python
from mlxtend.text import generalize_names
```


```python
generalize_names('Pozo, José Ángel')
```




    'pozo j'




```python
generalize_names('José Pozo')
```




    'pozo j'




```python
generalize_names('José Ángel Pozo')
```




    'pozo j'



## Example 1 - Optional Parameters


```python
from mlxtend.text import generalize_names
```


```python
generalize_names("Eto'o, Samuel", firstname_output_letters=2)
```




    'etoo sa'




```python
generalize_names("Eto'o, Samuel", firstname_output_letters=0)
```




    'etoo'




```python
generalize_names("Eto'o, Samuel", output_sep=', ')
```




    'etoo, s'



# API


```python
from mlxtend.text import generalize_names
help(generalize_names)
```

    Help on function generalize_names in module mlxtend.text.names:
    
    generalize_names(name, output_sep=' ', firstname_output_letters=1)
        Function that outputs a person's name in the format
        <last_name><separator><firstname letter(s)> (all lowercase)
        
        Parameters
        ----------
        name : `str`
          Name of the player
        
        output_sep : `str` (default: ' ')
          String for separating last name and first name in the output.
        
        firstname_output_letters : `int`
          Number of letters in the abbreviated first name.
        
        Returns
        ----------
        gen_name : `str`
          The generalized name.
    

