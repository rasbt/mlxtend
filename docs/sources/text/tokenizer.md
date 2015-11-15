
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



# Tokenizer

Different functions to tokenize text.

> from mlxtend.text import tokenizer_[type]

# Overview

Different functions to tokenize text.

### References

- -

### Related Topics

- [Name Generalization and Duplicates](./generalize_names_duplcheck.html)
- [Name Generalization](./generalize_names.html)

# Examples

## Example 1 - Extract Emoticons


```python
from mlxtend.text import tokenizer_emoticons
```


```python
tokenizer_emoticons('</a>This :) is :( a test :-)!')
```




    [':)', ':(', ':-)']



## Example 2 - Extract Words and Emoticons


```python
from mlxtend.text import tokenizer_words_and_emoticons
```


```python
tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!')
```




    ['this', 'is', 'a', 'test', ':)', ':(', ':-)']



# API


```python
from mlxtend.text import tokenizer_emoticons
help(tokenizer_emoticons)
```

    Help on function tokenizer_emoticons in module mlxtend.text.tokenizer:
    
    tokenizer_emoticons(text)
        Funtion that returns emoticons from text
        
        Example:
        >>> tokenizer_emoticons('</a>This :) is :( a test :-)!')
        [':)', ':(', ':-)']
    



```python
from mlxtend.text import tokenizer_words_and_emoticons
help(tokenizer_words_and_emoticons)
```

    Help on function tokenizer_words_and_emoticons in module mlxtend.text.tokenizer:
    
    tokenizer_words_and_emoticons(text)
        Funtion that returns lowercase words and emoticons from text
        
        Example:
        >>> tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!')
        ['this', 'is', 'a', 'test', ':)', ':(', ':-)']
    



```python

```
