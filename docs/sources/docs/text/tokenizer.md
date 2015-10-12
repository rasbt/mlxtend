mlxtend  
Sebastian Raschka, last updated: 05/20/2015


# Tokenizer


## Emoticons


> from mlxtend.text import extract_emoticons

A function that uses regular expressions to return a list ofemoticons from a text.


Example:

    >>> extract_emoticons('</a>This :) is :( a test :-)!')
    [':)', ':(', ':-)']


## Words and Emoticons


> from mlxtend.text import extract_words_and_emoticons

A function that uses regular expressions to return a list of words and emoticons from a text.


Example:

    >>> extract_words_and_emoticons('</a>This :) is :( a test :-)!')
    ['this', 'is', 'a', 'test', ':)', ':(', ':-)']
    
    
    
