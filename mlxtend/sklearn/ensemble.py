"""
Soft Voting/Majority Rule classifier

This module contains a Soft Voting/Majority Rule classifier for 
classification clfs.


"""
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ 
    Soft Voting/Majority Rule classifier for classification clfs.
        
    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of clfs for classification.
      
    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of 
      the sums of predict probalities.
    
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) that are multiplied with the 
      predicted class probabilities before averaging if `voting='soft'` 
      Uses uniform weights if `None`.
        
    Attributes
    ----------
    classes_ : array-like, shape = [n_class_labels]
       
    probas_ : array, shape = [n_probabilities, n_classifiers]
        Predicted probabilities by each classifier if `weights=array-like`.
        
    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB 
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> np.random.seed(123)
    >>> clf1 = LogisticRegression()
    >>> clf2 = RandomForestClassifier()
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
      
    """
    def __init__(self, clfs, voting='hard', weights=None):
        self.clfs = clfs
        
        if voting not in ('soft', 'hard'):
            raise ValueError("voting must be 'soft' or 'hard'; got (voting=%r)"
                             % voting)

        self.voting = voting
        
        if weights and len(weights) != len(clfs):
            raise ValueError('Number of classifiers and weights must be equal')      
        
        self.weights = weights
        self.le = LabelEncoder()
        self.classes_ = None


    def fit(self, X, y):
        """ 
        Fits the clfs.
        
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
            
        y : array-like, shape = [n_samples]
            Target values.
      
        Returns
        -------
        self : object

        """
        for clf in self.clfs:
            clf.fit(X, y)
        
        self.le.fit(y)
        self.classes_ = np.unique(y)
                
        return self
            
    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        
        Returns
        ----------
        maj : array-like, shape = [n_class_labels]
            Predicted class labels by majority rule.
        
        """
        if self.voting == 'soft':
            avg = self.predict_proba(X)
            maj = self.le.inverse_transform(np.argmax(avg, axis=1))
        
        else: # 'hard' voting
            self.classes_ = self._predict(X)
            
            # arange class labels as [n_class_labels, n_classifier_predictions]
            self.classes_ = np.asarray([self.classes_[:,c] for c in range(self.classes_.shape[1])])
            
            # duplicate class labels if weights are provided for argmax majority rule
            if self.weights:
                self.classes_ = np.concatenate([np.tile(self.classes_[:,c,None], w)
                                        for w,c in zip(self.weights, range(self.classes_.shape[1]))], axis=1)

            maj = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=1, arr=self.classes_)
        
        return maj
            
    
    def predict_proba(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.
        
        """
        self.probas_ = self._predict_probas(X)
        avg = np.average(self.probas_, axis=0, weights=self.weights)
        
        return avg

  
    def transform(self, X):
        """         
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
      
        Returns
        -------
        If not `weights=None`:
          array-like = [n_classifier_results, n_class_proba, n_class]
            Class probabilties calculated by each classifier.
        
        Else:
          array-like = [n_classifier_results, n_class_label]
            Class labels predicted by each classifier.
        
        """
        if self.weights:
            return self._predict_probas(X)
        else:
            return self._predict(X)  
    
    def _predict(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs])
        
    def _predict_probas(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs])
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()