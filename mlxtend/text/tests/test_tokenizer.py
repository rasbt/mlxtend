from mlxtend.text import tokenizer_words_and_emoticons
from mlxtend.text import tokenizer_emoticons


def test_tokenizer_words_and_emoticons_1():
    assert(tokenizer_words_and_emoticons('</a>This :) is :( a test :-)!') ==
           ['this', 'is', 'a', 'test', ':)', ':(', ':-)'])


def test_tokenizer_words_and_emoticons_2():
    assert(tokenizer_emoticons('</a>This :) is :( a test :-)!') ==
           [':)', ':(', ':-)'])
