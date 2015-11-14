import sys

if sys.version_info >= (3, 0):
    from .names import generalize_names
    from .names import generalize_names_duplcheck
    
from .tokenizer import extract_words_and_emoticons
from .tokenizer import extract_emoticons

__all__ = ["generalize_names", "generalize_names_duplcheck",
            "extract_words_and_emoticons", "extract_emoticons"]
