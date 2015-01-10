# Sebastian Raschka 08/13/2014
# mlxtend Machine Learning Library Extensions


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """ 
    Ensemble classifier for scikit-learn estimators.
        
    Parameters
    ----------
    
    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list`
      Weights for the individual classifiers for `.predict_proba`. 
      Using equal weight by default.
      
    """
    def __init__(self, clfs=None, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """ 
        Fit the scikit-learn estimators.
        
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels
      
        """
        for clf in self.clfs:
            clf.fit(X, y)
            
    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
        
        Returns
        ----------
        
        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule
        
        """
        
        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        
        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])
        
        return maj
            
    def predict_proba(self, X):
        
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
        
        Returns
        ----------
        
        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.
        
        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)
        
        return avg