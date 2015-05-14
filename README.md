[![Build Status](https://travis-ci.org/rasbt/mlxtend.svg?branch=dev)](https://travis-ci.org/rasbt/mlxtend)
[![PyPI version](https://badge.fury.io/py/mlxtend.svg)](http://badge.fury.io/py/mlxtend)
![Python 2.7](https://img.shields.io/badge/python-2.7-blue.svg)
![Python 3.4](https://img.shields.io/badge/python-3.4-blue.svg)
![License](https://img.shields.io/badge/license-BSD-blue.svg)

# mlxtend

**A library consisting of useful tools and extensions for the day-to-day data science tasks.**

<br>

Sebastian Raschka 2014-2015

Current version: 0.2.5

<br>


## Links
- Source code repository: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend)
- PyPI: [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend)






















<a id='pandas-utilities'></a>
## Pandas Utilities

[[back to top](#overview)]

The `pandas utilities` can be imported via

	from mxtend.pandas import ...


<br>
<br>
<a id='minmax-scaling'></a>
### Minmax Scaling

[[back to top](#overview)]

##### Description

A function that applies minmax scaling to pandas DataFrame columns.

- More information about the default parameters and additional options can be found [here](https://github.com/rasbt/mlxtend/blob/master/mlxtend/pandas/scaling.py#L5-28).

##### Examples

	from mlxtend.pandas import minmax_scaling

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/pandas_scaling_minmax_scaling.png)
    





<br>
<br>
<a id='standardizing'></a>
### Standardizing

[[back to top](#overview)]

##### Description

A function to standardize columns in pandas DataFrames so that they have properties of a standard normal distribution (mean=0, standard deviation=1).

- More information about the default parameters and additional options can be found [here](https://github.com/rasbt/mlxtend/blob/master/mlxtend/pandas/scaling.py#L40-57).

##### Examples

	from mlxtend.pandas import standardizing

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/pandas_scaling_standardizing.png)
    

	
<br>
<br>
<br>
<br>

<a id='file-io-utilities'></a>
## File IO Utilities

[[back to top](#overview)]

<br>

The `file_io utilities` can be imported via

	from mxtend.file_io import ...

<br>
<br>
<a id='find-files'></a>
### Find Files

[[back to top](#overview)]

##### Description

A function that finds files in a given directory based on substring matches and returns a list of the file names found.

##### Examples

	from mlxtend.file_io import find_files

    >>> find_files('mlxtend', '/Users/sebastian/Desktop')
	['/Users/sebastian/Desktop/mlxtend-0.1.6.tar.gz', 
	'/Users/sebastian/Desktop/mlxtend-0.1.7.tar.gz'] 
    

##### Default Parameters

    def find_files(substring, path, recursive=False, check_ext=None, ignore_invisible=True): 
        """
        Function that finds files in a directory based on substring matching.
        
        Parameters
        ----------
    
        substring : `str`
          Substring of the file to be matched.
    
        path : `str` 
          Path where to look.
    
        recursive: `bool`, optional, (default=`False`)
          If true, searches subdirectories recursively.
      
        check_ext: `str`, optional, (default=`None`)
          If string (e.g., '.txt'), only returns files that
          match the specified file extension.
      
        ignore_invisible : `bool`, optional, (default=`True`)
          If `True`, ignores invisible files (i.e., files starting with a period).
      
        Returns
        ----------
        results : `list`
          List of the matched files.
        
        """


<br>
<br>


<a id='find-file-groups'></a>
### Find File Groups

[[back to top](#overview)]

##### Description

A function that finds files that belong together (i.e., differ only by file extension) in different directories and collects them in a Python dictionary for further processing tasks. 

##### Examples

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/file_io_find_find_filegroups_1.png)

    d1 = os.path.join(master_path, 'dir_1')
    d2 = os.path.join(master_path, 'dir_2')
    d3 = os.path.join(master_path, 'dir_3')
    
    find_filegroups(paths=[d1,d2,d3], substring='file_1')
    # Returns:
    # {'file_1': ['/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_1/file_1.log', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_2/file_1.csv', 
    #             '/Users/sebastian/github/mlxtend/tests/data/find_filegroups/dir_3/file_1.txt']}
    #
    # Note: Setting `substring=''` would return a 
    # dictionary of all file paths for 
    # file_1.*, file_2.*, file_3.*

   
##### Default Parameters

    def find_filegroups(paths, substring='', extensions=None, validity_check=True, ignore_invisible=True):
        """
        Function that finds and groups files from different directories in a python dictionary.
        
        Parameters
        ----------
        paths : `list` 
          Paths of the directories to be searched. Dictionary keys are build from
          the first directory.
    
        substring : `str`, optional, (default=`''`)
          Substring that all files have to contain to be considered.
    
        extensions : `list`, optional, (default=`None`)
          `None` or `list` of allowed file extensions for each path. If provided, the number
          of extensions must match the number of `paths`.
         
        validity_check : `bool`, optional, (default=`True`)
          If `True`, checks if all dictionary values have the same number of file paths. Prints
          a warning and returns an empty dictionary if the validity check failed.

        ignore_invisible : `bool`, optional, (default=`True`)
          If `True`, ignores invisible files (i.e., files starting with a period).

        Returns
        ----------
        groups : `dict`
          Dictionary of files paths. Keys are the file names found in the first directory listed
          in `paths` (without file extension).
        
        """

<br>
<br>
<br>
<br>







<a id='scikit-learn-utilities'></a>

## Scikit-learn Utilities

[[back to top](#overview)]

<br>

The `scikit-learn utilities` can be imported via

	from mxtend.scikit-learn import ...

<br>
<br>

<a id='sequential-backward-selection'></a>
### Sequential Backward Selection

[[back to top](#overview)]

Sequential Backward Selection (SBS) is  a classic feature selection algorithm -- a greedy search algorithm -- that has been developed as a suboptimal solution to the computationally often not feasible exhaustive search. In a nutshell, SBS removes one feature at the time based on the classifier performance until a feature subset of the desired size *k* is reached. 

**Note that SBS is different from the [recursive feature elimination (RFE)](http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html#sklearn.feature_selection.RFE) that is implemented in scikit-learn.** RFE sequentially removes features based on the feature weights whereas SBS removes features based on the model performance.
More detailed explanations about the algorithms and examples can be found in [this IPython notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/sklearn_sequential_feature_select_sbs.ipynb).


##### Documentation

For more information about the parameters, please see the [mlxtend.sklearn.SBS](./mlxtend/sklearn/sequential_backward_select.py#L11-42) class documentation.


##### Examples

Input:

    from mlxtend.sklearn import SBS
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.datasets import load_iris

    iris = load_iris()
    X = iris.data
    y = iris.target

    knn = KNeighborsClassifier(n_neighbors=4)

    sbs = SBS(knn, k_features=2, scoring='accuracy', cv=5)
    sbs.fit(X, y)

    print('Indices of selected features:', sbs.indices_)
    print('CV score of selected subset:', sbs.k_score_)
    print('New feature subset:')
    sbs.transform(X)[0:5]

Output:

    Indices of selected features: (0, 3)
    CV score of selected subset: 0.96
    New feature subset:
    array([[ 5.1,  0.2],
       [ 4.9,  0.2],
       [ 4.7,  0.2],
       [ 4.6,  0.2],
       [ 5. ,  0.2]])
 
<br>
<br>

As demonstrated below, the SBS algorithm can be a useful alternative to dimensionality reduction techniques to reduce overfitting and where the original features need to be preserved:

    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

    scr = StandardScaler()
    X_std = scr.fit_transform(X)
 
    knn = KNeighborsClassifier(n_neighbors=4)
 
    # selecting features
    sbs = SBS(knn, k_features=1, scoring='accuracy', cv=5)
    sbs.fit(X_std, y)

    # plotting performance of feature subsets
    k_feat = [len(k) for k in sbs.subsets_]

    plt.plot(k_feat, sbs.scores_, marker='o')
    plt.ylabel('Accuracy')
    plt.xlabel('Number of features')
    plt.show()

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/sklearn_sequential_feature_select_sbs_wine_1.png)


<br>
More examples -- including how to use `SBS` in scikit-learn's `GridSearch` can be found in [this IPython notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/sklearn_sequential_feature_select_sbs.ipynb).






<br>
<br>
<br>
<br>

<a id='columnselector-for-custom-feature-selection'></a>
### ColumnSelector for Custom Feature Selection

[[back to top](#overview)]

A feature selector for scikit-learn's Pipeline class that returns specified columns from a NumPy array; extremely useful in combination with scikit-learn's `Pipeline` in cross-validation.

- [An example usage](http://nbviewer.ipython.org/github/rasbt/pattern_classification/blob/master/machine_learning/scikit-learn/scikit-pipeline.ipynb#Cross-Validation-and-Pipelines) of the `ColumnSelector` used in a pipeline for cross-validation on the Iris dataset.

Example in `Pipeline`:

	from mlxtend.sklearn import ColumnSelector
	from sklearn.pipeline import Pipeline
	from sklearn.naive_bayes import GaussianNB
	from sklearn.preprocessing import StandardScaler

	clf_2col = Pipeline(steps=[
	    ('scaler', StandardScaler()),
    	('reduce_dim', ColumnSelector(cols=(1,3))),    # extracts column 2 and 4
    	('classifier', GaussianNB())   
    	]) 

`ColumnSelector` has a `transform` method that is used to select and return columns (features) from a NumPy array so that it can be used in the `Pipeline` like other `transformation` classes. 

    ### original data
    
	print('First 3 rows before:\n', X_train[:3,:])
    First 3 rows before:
 	[[ 4.5  2.3  1.3  0.3]
 	[ 6.7  3.3  5.7  2.1]
 	[ 5.7  3.   4.2  1.2]]
	
	### after selection

	cols = ColumnExtractor(cols=(1,3)).transform(X_train)
	print('First 3 rows:\n', cols[:3,:])
	
	First 3 rows:
 	[[ 2.3  0.3]
 	[ 3.3  2.1]
 	[ 3.   1.2]]


<br>
<br>

<a id='densetransformer-for-pipelines-and-gridsearch'></a>
### DenseTransformer for Pipelines and GridSearch

[[back to top](#overview)]

A simple transformer that converts a sparse into a dense numpy array, e.g., required for scikit-learn's `Pipeline` when e.g,. `CountVectorizers` are used in combination with `RandomForest`s.


Example in `Pipeline`:

	from sklearn.pipeline import Pipeline
	from sklearn import metrics
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.feature_extraction.text import CountVectorizer

	from mlxtend.sklearn import DenseTransformer


	pipe_1 = Pipeline([
	    ('vect', CountVectorizer(analyzer='word',
	                      decode_error='replace',
	                      preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()), 
	                      stop_words=stopwords,) ),
	    ('to_dense', DenseTransformer()),
	    ('clf', RandomForestClassifier())
	])

	parameters_1 = dict(
	    clf__n_estimators=[50, 100, 200],
	    clf__max_features=['sqrt', 'log2', None],)

	grid_search_1 = GridSearchCV(pipe_1, 
	                           parameters_1, 
	                           n_jobs=1, 
	                           verbose=1,
	                           scoring=f1_scorer,
	                           cv=10)


	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipe_1.steps])
	print("parameters:")
	grid_search_1.fit(X_train, y_train)
	print("Best score: %0.3f" % grid_search_1.best_score_)
	print("Best parameters set:")
	best_parameters_1 = grid_search_1.best_estimator_.get_params()
	for param_name in sorted(parameters_1.keys()):
	    print("\t%s: %r" % (param_name, best_parameters_1[param_name]))


<br>
<br>

<a id='ensembleclassifier'></a>
### EnsembleClassifier

[[back to top](#overview)]

And ensemble classifier that predicts class labels based on a majority voting rule (hard voting) or average predicted probabilities (soft voting).

Decision regions plotted for 4 different classifiers:   

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/sklearn_ensemble_decsion_regions.png)

Please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/sklearn_ensemble_ensembleclassifier.ipynb) for a detailed explanation and examples.

##### Documentation

For more information about the parameters, please see the [`mlxtend.sklearn.EnsembleClassifier`](./mlxtend/sklearn/ensemble.py#L20-44) class documentation.

The `EnsembleClassifier` will likely be included in the scikit-learn library as `VotingClassifier` at some point, and during this implementation process, the `EnsembleClassifier` has been slightly improved based on valuable feedback from the scikit-learn community.

##### Examples

Input:

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

#####  GridSearch Example

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



<br>
<br>        
<br>
<br>
<a id='math-utilities'></a>
## Math Utilities

[[back to top](#overview)]

<br>

The `math utilities` can be imported via

	from mxtend.math import ...

<br>
<br>
<a id='combinations-and-permutations'></a>
### Combinations and Permutations

[[back to top](#overview)]

Functions to calculate the number of combinations and permutations for creating subsequences of *r* elements out of a sequence with *n* elements.

##### Examples

In:

	from mlxtend.math import num_combinations
	from mlxtend.math import num_permutations

	c = num_combinations(n=20, r=8, with_replacement=False)
	print('Number of ways to combine 20 elements into 8 subelements: %d' % c)

	d = num_permutations(n=20, r=8, with_replacement=False)
	print('Number of ways to permute 20 elements into 8 subelements: %d' % d)

Out:	

	Number of ways to combine 20 elements into 8 subelements: 125970
	Number of ways to permute 20 elements into 8 subelements: 5079110400

This is especially useful in combination with [`itertools`](https://docs.python.org/3/library/itertools.html), e.g., in order to estimate the progress via [`pyprind`](https://github.com/rasbt/pyprind).
    
    


![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/combinations_pyprind.png)

##### Default Parameters

    def num_combinations(n, r, with_replacement=False):
        """ 
        Function to calculate the number of possible combinations.
        
        Parameters
        ----------
        n : `int`
          Total number of items.
      
        r : `int`
          Number of elements of the target itemset.
    
        with_replacement : `bool`, optional, (default=False)
          Allows repeated elements if True.
      
        Returns
        ----------
        comb : `int`
          Number of possible combinations.
        
        """


    def num_permutations(n, r, with_replacement=False):
        """ 
        Function to calculate the number of possible permutations.
        
        Parameters
        ----------
        n : `int`
          Total number of items.
    
        r : `int`
          Number of elements of the target itemset.
    
        with_replacement : `bool`, optional, (default=False)
          Allows repeated elements if True.
      
        Returns
        ----------
        permut : `int`
          Number of possible permutations.
        
        """
   

<br>
<br>        
<br>
<br>
<a id='matplotlib-utilities'></a>
## Matplotlib Utilities

[[back to top](#overview)]

<br>

The `matplotlib utilities` can be imported via

	from mxtend.matplotlib import ...

<br>
<br>
<a id='stacked-barplot'></a>
### Stacked Barplot

A function to conveniently plot stacked bar plots in matplotlib using pandas `DataFrame`s. 

Please see the code implementation for the [default parameters](./mlxtend/matplotlib/stacked_barplot.py#L5-38).

<br>
#### Example

Creating an example  `DataFrame`:	
	
    import pandas as pd

    s1 = [1.0, 2.0, 3.0, 4.0]
	s2 = [1.4, 2.1, 2.9, 5.1]
	s3 = [1.9, 2.2, 3.5, 4.1]
	s4 = [1.4, 2.5, 3.5, 4.2]
	data = [s1, s2, s3, s4]
	
	df = pd.DataFrame(data, columns=['X1', 'X2', 'X3', 'X4'])
	df.columns = ['X1', 'X2', 'X3', 'X4']
	df.index = ['Sample1', 'Sample2', 'Sample3', 'Sample4']
	df
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_stacked_barplot_1.png)
	
Plotting the stacked barplot. By default, the index of the `DataFrame` is used as column labels, and the `DataFrame` columns are used for the plot legend.

	from mlxtend.matplotlib import stacked_barplot

	stacked_barplot(df, rotation=45)
	
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_stacked_barplot_2.png)
	


<br>
<br>
<a id='enrichment-plot'></a>
### Enrichment Plot

A function to plot step plots of cumulative counts.

Please see the code implementation for the [default parameters](./mlxtend/matplotlib/enrichment_plot.py#L5-48).

<br>
#### Example

Creating an example  `DataFrame`:	
	
    import pandas as pd
    s1 = [1.1, 1.5]
    s2 = [2.1, 1.8]
    s3 = [3.1, 2.1]
    s4 = [3.9, 2.5]
    data = [s1, s2, s3, s4]
    df = pd.DataFrame(data, columns=['X1', 'X2'])
    df
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_enrichment_plot_1.png)
	
Plotting the enrichment plot. The y-axis can be interpreted as "how many samples are less or equal to the corresponding x-axis label."

    from mlxtend.matplotlib import enrichment_plot
    enrichment_plot(df)
	
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_enrichment_plot_2.png)
	



<br>
<br>
<a id='category-scatter'></a>
### Category Scatter

A function to quickly produce a scatter plot colored by categories from a pandas `DataFrame` or NumPy `ndarray` object.

Please see the implementation for the [default parameters](./mlxtend/matplotlib/scatter.py#L6-42).

<br>
#### Example

Loading an example dataset as pandas `DataFrame`:	
	
	import pandas as pd

	df = pd.read_csv('/Users/sebastian/Desktop/data.csv')
	df.head()
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_categorical_scatter_1.png)
	
Plotting the data where the categories are determined by the unique values in the label column `label_col`. The `x` and `y` values are simply the column names of the DataFrame that we want to plot.

	import matplotlib.pyplot as plt
	from mlxtend.matplotlib import category_scatter

	category_scatter(x='x', y='y', label_col='label', data=df)
           
	plt.legend(loc='best')
	
	
![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_categorical_scatter_2.png)
	

Similarly, we can also use NumPy arrays. E.g.,

	X = 

	array([['class1', 10.0, 8.04],
       ['class1', 8.0, 6.95],
       ['class1', 13.2, 7.58],
       ['class1', 9.0, 8.81],
		...
       ['class4', 8.0, 5.56],
       ['class4', 8.0, 7.91],
       ['class4', 8.0, 6.89]], dtype=object)
       
Where the `x`, `y`, and `label_col` refer to the respective column indices in the array:

	category_scatter(x=1, y=2, label_col=0, data=df.values)
           
	plt.legend(loc='best')

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/matplotlib_categorical_scatter_2.png)

<br>
<br>
<a id='removing-borders'></a>
### Removing Borders

[[back to top](#overview)]

A function to remove borders from `matplotlib` plots. Import `remove_borders` via

    from mlxtend.matplotlib import remove_borders




	def remove_borders(axes, left=False, bottom=False, right=True, top=True):
    	""" 
    	A function to remove chartchunk from matplotlib plots, such as axes
        	spines, ticks, and labels.
        
        	Keyword arguments:
            	axes: An iterable containing plt.gca() or plt.subplot() objects, e.g. [plt.gca()].
            	left, bottom, right, top: Boolean to specify which plot axes to hide.
            
    	"""

##### Examples

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/remove_borders_3.png)

<br>
<br>

