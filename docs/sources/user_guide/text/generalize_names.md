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


*generalize_names(name, output_sep=' ', firstname_output_letters=1)*

Generalize a person's first and last name.

Returns a person's name in the format
    `<last_name><separator><firstname letter(s)> (all lowercase)`

**Parameters**

- `name` : `str`

    Name of the player

- `output_sep` : `str` (default: ' ')

    String for separating last name and first name in the output.

- `firstname_output_letters` : `int`

    Number of letters in the abbreviated first name.

**Returns**

- `gen_name` : `str`

    The generalized name.


