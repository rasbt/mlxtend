## generalize_names

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

