mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>

<div style="width:500px;height:70px;border:1px solid #000;padding:10px;background-color:#e5ffe5;"><p>If you are interested in using the <code>EnsembleClassifier</code>, please note that it is now also available through scikit learn (>0.17) as <a href="http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html"><code>VotingClassifier</code></a>.</p></div>


# Majority Rule Ensemble Classifier

> from mlxtend.classifier import EnsembleClassifier

And ensemble classifier (for scikit-learn estimators) that predicts class labels based on a majority voting rule (hard voting) or average predicted probabilities (soft voting).

Decision regions plotted for 4 different classifiers:   

![](./img/classifier_ensemble_decsion_regions.png)

Please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/sklearn_ensemble_ensembleclassifier.ipynb) for a detailed explanation and examples.


The `EnsembleClassifier` will likely be included in the scikit-learn library as `VotingClassifier` at some point, and during this implementation process, the `EnsembleClassifier` has been slightly improved based on valuable feedback from the scikit-learn community.

<hr>

## Cross-validation Example

Input:

	from mlxtend.classifier import EnsembleClassifier
	from sklearn import cross_validation
	from sklearn.linear_model import LogisticRegression
	from sklearn.naive_bayes import GaussianNB
	from sklearn.ensemble import RandomForestClassifier
	import numpy as np
	from sklearn import datasets

	iris = datasets.load_iris()
	X, y = iris.data[:, 1:3], iris.target

	np.random.seed(123)

    ################################
    # Initialize classifiers
    ################################

	clf1 = LogisticRegression()
	clf2 = RandomForestClassifier()
	clf3 = GaussianNB()

    ################################
    # Initialize EnsembleClassifier
    ################################

    # hard voting    
	eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='hard')

    # soft voting (uniform weights)
    # eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')

    # soft voting with different weights
    # eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft', weights=[1,2,10])



    ################################
    # 5-fold Cross-Validation
    ################################

	for clf, label in zip([clf1, clf2, clf3, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'Ensemble']):

	    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
	    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

Output:

	Accuracy: 0.90 (+/- 0.05) [Logistic Regression]
	Accuracy: 0.92 (+/- 0.05) [Random Forest]
	Accuracy: 0.91 (+/- 0.04) [naive Bayes]
	Accuracy: 0.95 (+/- 0.05) [Ensemble]

<br>
<br>

##  GridSearch Example

The `EnsembleClassifier` van also be used in combination with scikit-learns gridsearch module:


	from sklearn.grid_search import GridSearchCV

	clf1 = LogisticRegression(random_state=1)
	clf2 = RandomForestClassifier(random_state=1)
	clf3 = GaussianNB()
	eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], voting='soft')

	params = {'logisticregression__C': [1.0, 100.0],
          'randomforestclassifier__n_estimators': [20, 200],}

	grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
	grid.fit(iris.data, iris.target)

	for params, mean_score, scores in grid.grid_scores_:
    	print("%0.3f (+/-%0.03f) for %r"
            % (mean_score, scores.std() / 2, params))

Output:

	0.953 (+/-0.013) for {'randomforestclassifier__n_estimators': 20, 'logisticregression__C': 1.0}
	0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 200, 'logisticregression__C': 1.0}
	0.960 (+/-0.012) for {'randomforestclassifier__n_estimators': 20, 'logisticregression__C': 100.0}
	0.953 (+/-0.017) for {'randomforestclassifier__n_estimators': 200, 'logisticregression__C': 100.0}


**Note**:

If the `EnsembleClassifier` is initialized with multiple similar estimator objects, the estimator names are modified with consecutive integer indices, for example:


    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    eclf = EnsembleClassifier(clfs=[clf1, clf1, clf2], voting='soft')

    params = {'logisticregression-1__C': [1.0, 100.0],
              'logisticregression-2__C': [1.0, 100.0],
              'randomforestclassifier__n_estimators': [20, 200],}

    grid = GridSearchCV(estimator=eclf, param_grid=params, cv=5)
    grid.fit(iris.data, iris.target)


<hr>

## Default Parameters
<pre>class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """ Soft Voting/Majority Rule classifier for unfitted clfs.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of classifiers.
      Invoking the `fit` method on the `VotingClassifier` will fit clones
      of those original classifiers that will be stored in the class attribute
      `self.clfs_`.

    voting : str, {'hard', 'soft'} (default='hard')
      If 'hard', uses predicted class labels for majority rule voting.
      Else if 'soft', predicts the class label based on the argmax of
      the sums of the predicted probalities, which is recommended for
      an ensemble of well-calibrated classifiers.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      Sequence of weights (`float` or `int`) to weight the occurances of
      predicted class labels (`hard` voting) or class probabilities
      before averaging (`soft` voting). Uses uniform weights if `None`.

    verbose : int, optional (default=0)
      Controls the verbosity of the building process.
        `verbose=0` (default): Prints nothing
        `verbose=1`: Prints the number & name of the clf being fitted
        `verbose=2`: Prints info about the parameters of the clf being fitted
        `verbose>2`: Changes `verbose` param of the underlying clf to self.verbose - 2

    Attributes
    ----------
    classes_ : array-like, shape = [n_predictions]

    clf : array-like, shape = [n_predictions]
      The unmodified input classifiers

    clf_ : array-like, shape = [n_predictions]
      Fitted clones of the input classifiers

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf1 = LogisticRegression(random_state=1)
    >>> clf2 = RandomForestClassifier(random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> eclf2 = VotingClassifier(clfs=[clf1, clf2, clf3], voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]
    >>> eclf3 = VotingClassifier(clfs=[clf1, clf2, clf3],
    ...                          voting='soft', weights=[2,1,1])
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>>
    """</pre>


<hr>
## Methods

<pre>    def fit(self, X, y):
        """ Fit the clfs.

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
        """</pre>


<pre>    def predict(self, X):
        """ Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """</pre>

<pre>predict_proba(self, X):
        """ Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """</pre>


<pre>    def transform(self, X):
        """ Return class labels or probabilities for X for each estimator.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilties calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_classifiers, n_samples]
            Class labels predicted by each classifier.
        """</pre>
