from mlxtend.text import extract_words_and_emoticons
from mlxtend.text import extract_emoticons

def test_extract_words_and_emoticons():
    assert(extract_words_and_emoticons('</a>This :) is :( a test :-)!') == ['this', 'is', 'a', 'test', ':)', ':(', ':-)'])
    
def test_extract_words_and_emoticons():
    assert(extract_emoticons('</a>This :) is :( a test :-)!') == [':)', ':(', ':-)'])