# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Functions for tokenizing text data.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause

import re


def tokenizer_words_and_emoticons(text):
    """Convert text to lowercase words and emoticons.

    Example:
    >>> tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!')
    ['this', 'is', 'a', 'test', ':)', ':(', ':-)']
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons)
    return text.split()


def tokenizer_emoticons(text):
    """Return emoticons from text

    Example:
    >>> tokenizer_emoticons('</a>This :) is :( a test :-)!')
    [':)', ':(', ':-)']
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    return emoticons
