# Sebastian Raschka 2014-2026
# mlxtend Machine Learning Library Extensions
#
# SVM-based sentiment classification for text data.
#
# License: BSD 3 clause

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.utils.validation import check_is_fitted


class SentimentSVM(BaseEstimator, ClassifierMixin):
    """TF-IDF text classifier using a linear Support Vector Machine.

    This estimator converts text documents into TF-IDF features and trains
    a linear SVM classifier. It can be used for binary or multiclass
    classification, including positive, negative, and neutral sentiment.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter for the SVM. Larger values allow the model
        to fit the training data more closely.

    ngram_range : tuple, default=(1, 1)
        Lower and upper boundary of the n-grams used by the TF-IDF vectorizer.

    probability : bool, default=False
        Whether to enable probability estimates. Enabling this option makes
        model fitting slower.

    random_state : int or None, default=None
        Controls the randomness used when probability estimates are enabled.

    Attributes
    ----------
    pipeline_ : sklearn.pipeline.Pipeline
        Fitted TF-IDF and SVM pipeline.

    classes_ : ndarray
        Class labels learned during fitting.

    Examples
    --------
    >>> texts = [
    ...     "I love this product",
    ...     "This product is terrible",
    ...     "The product arrived today",
    ... ]
    >>> labels = ["positive", "negative", "neutral"]
    >>> classifier = SentimentSVM()
    >>> classifier.fit(texts, labels)
    SentimentSVM()
    >>> classifier.predict(["I really love it"])
    array(['positive'], dtype='<U8')
    """

    def __init__(
        self,
        C=1.0,
        ngram_range=(1, 1),
        probability=False,
        random_state=None,
    ):
        self.C = C
        self.ngram_range = ngram_range
        self.probability = probability
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the TF-IDF vectorizer and SVM classifier.

        Parameters
        ----------
        X : array-like of str
            Training text documents.

        y : array-like
            Target labels.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.pipeline_ = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(ngram_range=self.ngram_range),
                ),
                (
                    "svm",
                    SVC(
                        kernel="linear",
                        C=self.C,
                        probability=self.probability,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.pipeline_.fit(X, y)
        self.classes_ = self.pipeline_.named_steps["svm"].classes_

        return self

    def predict(self, X):
        """Predict labels for text documents.

        Parameters
        ----------
        X : array-like of str
            Text documents to classify.

        Returns
        -------
        ndarray
            Predicted labels.
        """
        check_is_fitted(self, ["pipeline_", "classes_"])
        return self.pipeline_.predict(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of str
            Test text documents.

        y : array-like
            True labels.

        Returns
        -------
        float
            Mean accuracy.
        """
        check_is_fitted(self, ["pipeline_", "classes_"])
        return self.pipeline_.score(X, y)
