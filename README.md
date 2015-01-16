# mlxtend


A library of Python tools and extensions for data science.

Link to the `mlxtend` repository on GitHub: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend).

<br>

Sebastian Raschka 2014

<br>
<br>

<a id='overview'></a>
## Overview

- [Preprocessing](#preprocessing)
	- [MeanCenterer](#meancenterer) 
- [Text Utilities](#text-utilities)
	- [Name Generalization](#name-generelization)
	- [Name Generalization and Duplicates](#name-generalization-and-duplicates)
- [File IO Utilities](#file-io-utilities)
	- [Find Files](#find-files)
	- [Find File Groups](#find-file-groups)
- [Scikit-learn Utilities](#scikit-learn-utilities)
	- [ColumnSelector for Custom Feature Selection](#columnselector-for-custom-feature-selection) 
	- [DenseTransformer for Pipelines and GridSearch](#densetransformer-for-pipelines-and-gridsearch)
	- [EnsembleClassifier to Combine Classification Models](#ensembleclassifier) 
- [Math Utilities](#math-utilities)
	- [Combinations and Permutations](#combinations-and-permutations)
- [Matplotlib Utilities](#matplotlib-utilities)
	- [Removing Borders](#remove_borders) 
- [Installation](#installation)
- [Changelog](https://github.com/rasbt/mlxtend/blob/master/docs/CHANGELOG.txt)


<br>
<br>
<br>
<br>

<a id='preprocessing'></a>
## Preprocessing

[[back to top](#overview)]

A collection of different functions for various data preprocessing procedures.

The `preprocessing utilities` can be imported via

	from mxtend.preprocessing import ...
	
<br>
<br>
<a id='meancenterer'></a>
### MeanCenterer

[[back to top](#overview)]

	class MeanCenterer(TransformerObj):
    """
    Class for column centering of vectors and matrices.
    
    Keyword arguments:
        X: NumPy array object where each attribute/variable is
           stored in an individual column. 
           Also accepts 1-dimensional Python list objects.
    
    Class methods:
        fit: Fits column means to MeanCenterer object.
        transform: Uses column means from `fit` for mean centering.
        fit_transform: Fits column means and performs mean centering.
    
    The class methods `transform` and `fit_transform` return a new numpy array
    object where the attributes are centered at the column means.
    
    """
<br>
    
**Examples:**

Use the `fit` method to fit the column means of a dataset (e.g., the training dataset) to a new `MeanCenterer` object. Then, call the `transform` method on the same dataset to center it at the sample mean.

	>>> X_train
	array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
    >>> mc = MeanCenterer().fit(X_train)
	>>> mc.transform(X_train)
    array([[-3, -3, -3],
       [ 0,  0,  0],
       [ 3,  3,  3]])

<br>

To use the same parameters that were used to center the training dataset, simply call the `transform` method of the `MeanCenterer` instance on a new dataset (e.g., test dataset).
    
    >>> X_test 
    array([[1, 1, 1],
       [1, 1, 1],
       [1, 1, 1]])
    >>> mc.transform(X_test)  
    array([[-3, -4, -5],
       [-3, -4, -5],
       [-3, -4, -5]])

<br>

The `MeanCenterer` also supports Python list objects, and the `fit_transform` method allows you to directly fit and center the dataset.

	>>> Z
	[1, 2, 3]
	>>> MeanCenterer().fit_transform(Z)
	array([-1,  0,  1])


<br>

	import matplotlib.pyplot as plt
	import numpy as np

	X = 2 * np.random.randn(100,2) + 5

	plt.scatter(X[:,0], X[:,1])
	plt.grid()
	plt.title('Random Gaussian data w. mean=5, sigma=2')
	plt.show()

	Y = MeanCenterer.fit_transform(X)
	plt.scatter(Y[:,0], Y[:,1])
	plt.grid()
	plt.title('Data after mean centering')
	plt.show()

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/mean_centering_3.png)


<br>
<br>
<br>
<br>




<a id='text-utilities'></a>

## Text Utilities

[[back to top](#overview)]

<br>

The `text utilities` can be imported via

	from mxtend.text import ...

<br>
<br>

<a id='name-generalization'></a>
### Name Generalization

[[back to top](#overview)]

##### Description

A function that converts a name into a general format ` <last_name><separator><firstname letter(s)> (all lowercase)`, which is useful if data is collected from different sources and is supposed to be compared or merged based on name identifiers. E.g., if names are stored in a pandas `DataFrame` column, the apply function can be used to generalize names: `df['name'] = df['name'].apply(generalize_names)`

##### Examples

	from mlxtend.text import generalize_names

    # defaults
    >>> generalize_names('Pozo, José Ángel')
    'pozo j'
    >>> generalize_names('Pozo, José Ángel') 
    'pozo j'
    >>> assert(generalize_names('José Ángel Pozo') 
    'pozo j' 
    >>> generalize_names('José Pozo')
    'pozo j' 
    
    # optional parameters
    >>> generalize_names("Eto'o, Samuel", firstname_output_letters=2)
    'etoo sa'
    >>> generalize_names("Eto'o, Samuel", firstname_output_letters=0)
    'etoo'
    >>> generalize_names("Eto'o, Samuel", output_sep=', ')
    'etoo, s' 

##### Default Parameters

	def generalize_names(name, output_sep=' ', firstname_output_letters=1):
	    """
	    Function that outputs a person's name in the format 
	    <last_name><separator><firstname letter(s)> (all lowercase)
	        
	    Parameters
	    ----------
	    name : `str`
	      Name of the player
	    output_sep : `str` (default: ' ')
	      String for separating last name and first name in the output.
	    firstname_output_letters : `int`
	      Number of letters in the abbreviated first name.
	      
	    Returns
	    ----------
	    gen_name : `str`
	      The generalized name.
	        
	    """

<br>
<br>

<a id='name-generalization-and-duplicates'></a>
### Name Generalization and Duplicates

[[back to top](#overview)]

**Note** that using [`generalize_names`](#name-generalization) with few `firstname_output_letters` can result in duplicate entries. E.g., if your dataset contains the names "Adam Johnson" and "Andrew Johnson", the default setting (i.e., 1 first name letter) will produce the generalized name "johnson a" in both cases.

One solution is to increase the number of first name letters in the output by setting the parameter `firstname_output_letters` to a value larger than 1. 

An alternative solution is to use the `generalize_names_duplcheck` function if you are working with pandas DataFrames. 

The  `generalize_names_duplcheck` function can be imported via

	from mlxtend.text import generalize_names_duplcheck

By default,  `generalize_names_duplcheck` will apply  `generalize_names` to a pandas DataFrame column with the minimum number of first name letters and append as many first name letters as necessary until no duplicates are present in the given DataFrame column. An example dataset column that contains the names  


##### Examples



Reading in a CSV file that has column `Name` for which we want to generalize the names:

- Samuel Eto'o
- Adam Johnson
- Andrew Johnson

<br>

    df = pd.read_csv(path)


Applying `generalize_names_duplcheck` to generate a new DataFrame with the generalized names without duplicates:	      

    df_new = generalize_names_duplcheck(df=df, col_name='Name')
- etoo s
- johnson ad
- jonson an

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

And ensemble classifier that predicts class labels based on a majority voting rule.

Please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/sklearn_ensemble_ensembleclassifier.ipynb) for a detailed explanation and examples.

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
    
	eclf = EnsembleClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

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
<a id='remove_borders'></a>
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

<a id='installation'></a>
## Installation

[[back to top](#overview)]

You can use the following command to install `mlxtend`:  
`pip install mlxtend`  
 or    
`easy_install mlxtend`  

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

`python setup.py install`  