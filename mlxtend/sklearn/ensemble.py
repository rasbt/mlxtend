# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions

"""
Soft Voting/Majority Rule classifier

This module contains a Soft Voting/Majority Rule ensemble classifier for 
scikit-learn classification estimators.

## Examples:

Intializing different scikit-learn estimators for classification
>>> clf1 = LogisticRegression()
>>> clf2 = RandomForestClassifier()
>>> clf3 = GaussianNB()

Example 1: Using Majority Class Label
initialization for using the majority class labels for prediction
>>> eclf = VotingClassifier(clfs=[clf1, clf2, clf3], voting='hard')
>>> eclf.fit(X_train, y_train)
>>> eclf.fit(X_test)

Example 2: Using Average Probabilities
initialization for using average predict_proba scores 
to derive the class label.
>>> eclf = VotingClassifier(clfs=[clf1, clf2, clf3], voting='soft')
>>> eclf.fit(X_train, y_train)
>>> eclf.fit(X_test)

Example 3: Using Weighted Average Probabilities
initialization for multiplying average predict_proba 
scores to derive the class label.
>>> eclf = VotingClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[2,1,5])
>>> eclf.fit(X_train, y_train)
>>> eclf.fit(X_test)

"""



from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ 
    Soft Voting/Majority Rule ensemble classifier for scikit-learn estimators.
        
    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of scikit-learn classifier objects.
      
    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of 
      the sums of predict probalities.
    
    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) that are multiplied with the 
      predicted class probabilities before averaging if `voting='soft'` 
      (weights are ignored if `voting='hard'`).
      Assumes equal weights if `None`.
        
    Attributes
    ----------
    classes_ : array-like, shape = [n_class_labels, n_classifiers]
        Class labels predicted by each classifier if `weights=None`.
       
    probas_ : array, shape = [n_probabilities, n_classifiers]
        Predicted probabilities by each classifier if `weights=array-like`.
      
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

    def fit(self, X, y):
        """ 
        Fits the scikit-learn estimators.
        
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
            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        
        else: # 'hard' voting
            self.classes_ = self._get_classes(X)
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])]) 
        
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
        self.probas_ = self._get_probas(X)
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
            return self._get_probas(X)
        else:
            return self._get_classes(X)  
    
    def _get_classes(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs])
        
    def _get_probas(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self.clfs])