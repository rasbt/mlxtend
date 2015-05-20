"""
Functions to tokenize text.

"""

# mlxtend
# Author: 
#        Sebastian Raschka

import re

def extract_words_and_emoticons(text):
    """Funtion that returns lowercase words and emoticons from text
    
    Example:
    >>> extract_words_and_emoticons('</a>This :) is :( a test :-)!')
    ['this', 'is', 'a', 'test', ':)', ':(', ':-)']
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons)
    return text.split()

    
def extract_emoticons(text):
    """Funtion that returns emoticons from text
    
    Example:
    >>> extract_emoticons('</a>This :) is :( a test :-)!')
    [':)', ':(', ':-)']
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    return emoticons