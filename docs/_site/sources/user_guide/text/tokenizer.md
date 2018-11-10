# Tokenizer

Different functions to tokenize text.

> from mlxtend.text import tokenizer_[type]

## Overview

Different functions to tokenize text for natural language processing tasks, for example such as building a bag-of-words model for text classification.

### References

- -

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



## API


*tokenizer_emoticons(text)*

Return emoticons from text

**Examples**


    >>> tokenizer_emoticons('</a>This :) is :( a test :-)!')
[':)', ':(', ':-)']

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_emoticons/](http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_emoticons/)

<br><br>
*tokenizer_words_and_emoticons(text)*

Convert text to lowercase words and emoticons.

**Examples**


    >>> tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!')
['this', 'is', 'a', 'test', ':)', ':(', ':-)']

For more usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_words_and_emoticons/](http://rasbt.github.io/mlxtend/user_guide/text/tokenizer_words_and_emoticons/)


