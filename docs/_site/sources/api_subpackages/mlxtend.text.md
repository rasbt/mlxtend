mlxtend version: 0.15.0dev 
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

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/text/generalize_names/](http://rasbt.github.io/mlxtend/user_guide/text/generalize_names/)




## generalize_names_duplcheck

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




## tokenizer_emoticons

*tokenizer_emoticons(text)*

Return emoticons from text

**Examples**


    >>> tokenizer_emoticons('</a>This :) is :( a test :-)!')
[':)', ':(', ':-)']

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_emoticons/](http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_emoticons/)




## tokenizer_words_and_emoticons

*tokenizer_words_and_emoticons(text)*

Convert text to lowercase words and emoticons.

**Examples**


    >>> tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!')
['this', 'is', 'a', 'test', ':)', ':(', ':-)']

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_words_and_emoticons/](http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_words_and_emoticons/)




