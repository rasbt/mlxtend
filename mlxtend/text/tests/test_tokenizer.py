import pytest

from mlxtend.text import tokenizer_emoticons, tokenizer_words_and_emoticons


def test_tokenizer_words_and_emoticons_1():
    assert tokenizer_words_and_emoticons("</a>This :) is :( a test :-)!") == [
        "this",
        "is",
        "a",
        "test",
        ":)",
        ":(",
        ":-)",
    ]


def test_tokenizer_words_and_emoticons_2():
    assert tokenizer_emoticons("</a>This :) is :( a test :-)!") == [":)", ":(", ":-)"]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Hi :) world", ["hi", "world", ":)"]),
        (":) happy", ["happy", ":)"]),
        ("test :( sad", ["test", "sad", ":("]),
        ("<b>:) hello</b>", ["hello", ":)"]),
        ("A :) B :( C", ["a", "b", "c", ":)", ":("]),
        ("x :-) y", ["x", "y", ":-)"]),
        ("Hello world", ["hello", "world"]),
        ("Hello :)", ["hello", ":)"]),
        ("", []),
        (
            "</a>This :) is :( a test :-)!",
            ["this", "is", "a", "test", ":)", ":(", ":-)"],
        ),
    ],
)
def test_tokenization(text, expected):
    assert tokenizer_words_and_emoticons(text) == expected


def test_emoticons_only_unchanged():
    assert tokenizer_emoticons("Hi :) world :(") == [":)", ":("]
