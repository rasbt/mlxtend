import pytest
from sklearn.exceptions import NotFittedError

from mlxtend.text import SentimentSVM

TRAIN_TEXTS = [
    "excellent wonderful amazing",
    "great enjoyable love",
    "happy satisfied good",
    "terrible awful horrible",
    "bad disappointing hate",
    "poor frustrating worst",
    "ordinary average okay",
    "normal routine standard",
    "neutral unchanged typical",
]

TRAIN_LABELS = [
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
]


def test_sentiment_svm_fit_and_predict():
    classifier = SentimentSVM()
    classifier.fit(TRAIN_TEXTS, TRAIN_LABELS)

    predictions = classifier.predict(
        [
            "excellent wonderful amazing",
            "terrible awful horrible",
            "ordinary average okay",
        ]
    )

    assert predictions.tolist() == ["positive", "negative", "neutral"]


def test_sentiment_svm_supports_multiclass_labels():
    classifier = SentimentSVM()
    classifier.fit(TRAIN_TEXTS, TRAIN_LABELS)

    predictions = classifier.predict(["great enjoyable love"])

    assert predictions.tolist() == ["positive"]
    assert set(classifier.classes_) == {"positive", "negative", "neutral"}


def test_sentiment_svm_score():
    classifier = SentimentSVM()
    classifier.fit(TRAIN_TEXTS, TRAIN_LABELS)

    assert classifier.score(TRAIN_TEXTS, TRAIN_LABELS) == pytest.approx(1.0)


def test_sentiment_svm_requires_fitting_before_prediction():
    classifier = SentimentSVM()

    with pytest.raises(NotFittedError):
        classifier.predict(["This is a test"])


def test_sentiment_svm_requires_fitting_before_scoring():
    classifier = SentimentSVM()

    with pytest.raises(NotFittedError):
        classifier.score(["This is a test"], ["neutral"])
