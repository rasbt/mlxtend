# mlxtend


A library of Python tools and extensions for data science.

Link to the `mlxtend` repository on GitHub: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend).

<br>

Sebastian Raschka 2014-2015

Current version: 0.2.5

<br>
<br>

<a id='overview'></a>
## Overview

- [Evaluate](#evaluate)
	- [Plotting Decision Regions](#plotting_decision_regions) 
	- [Plotting Learning Curves](#plotting_learning_curves)
- [Classifier](#classifier)
	- [Perceptron](#perceptron) 
	- [Adaline](#adaline) 
	- [Logistic Regression](#logistic-regression) 
- [Preprocessing](#preprocessing)
	- [MeanCenterer](#meancenterer) 
	- [Array Unison Shuffling](#array-unison-shuffling)
- [Regression](#regression)
	- [Plotting Linear Regression Fits](plotting-linear-regression-fits)
- [Text Utilities](#text-utilities)
	- [Name Generalization](#name-generalization)
	- [Name Generalization and Duplicates](#name-generalization-and-duplicates)
- [Pandas Utilities](#pandas-utilities)
	- [Minmax Scaling](#minmax-scaling)
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
	- [Category Scatter](#category-scatter) 
	- [Removing Borders](#removing-borders) 
- [Installation](#installation)
- [Changelog](https://github.com/rasbt/mlxtend/blob/master/docs/CHANGELOG.txt)


<br>
<br>
<br>
<br>

<a id='evaluate'></a>
## Evaluate

<br>
<br>
<a id='plotting_decision_regions'></a>
### Plotting Decision Regions

[[back to top](#overview)]


##### Description


- A function to plot decision regions of classifiers.  

- Import `plot_decision_regions` via

    from mlxtend.evaluate import plot_decision_regions

- Please see the implementation for the [default parameters](./mlxtend/evaluate/decision_regions.py#L11-64).




<br>
<br>

##### Examples

For more examples, please see this [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/evaluate_decision_regions.ipynb).




##### 2D example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_2d.png)

	from mlxtend.evaluate import plot_decision_regions
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.svm import SVC

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data[:, [0,2]]
	y = iris.target

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X,y)

	# Plotting decision regions
	plot_decision_regions(X, y, clf=svm, res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('SVM on Iris')
	plt.show()

##### 1D example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_1d.png)

	from mlxtend.evaluate import plot_decision_regions
	import matplotlib.pyplot as plt
	from sklearn import datasets
	from sklearn.svm import SVC

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data[:, 2]
	X = X[:, None]
	y = iris.target

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X,y)

	# Plotting decision regions
	plot_decision_regions(X, y, clf=svm, res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [cm]')
	plt.ylabel('petal length [cm]')
	plt.title('SVM on Iris')
	plt.show()

##### Highlighting Test Data Points

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_decision_regions_highlight.png)

Via the `X_highlight`, a second dataset can be provided to highlight particular points in the dataset via a circle.

	from sklearn.cross_validation import train_test_split

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0) 

	# Training a classifier
	svm = SVC(C=0.5, kernel='linear')
	svm.fit(X_train, y_train)

	# Plotting decision regions

	plot_decision_regions(X, y, clf=svm, 
                      X_highlight=X_test, 
                      res=0.02, legend=2)

	# Adding axes annotations
	plt.xlabel('sepal length [standardized]')
	plt.ylabel('petal length [standardized]')
	plt.title('SVM on Iris')
	plt.show()
<br>
<br>


<a id='plotting_learning_curves'></a>
### Plotting Learning Curves
[[back to top](#overview)]


##### Description


A function to plot learning curves for classifiers. Learning curves are extremely useful to analyze if a model is suffering from over- or under-fitting (high variance or high bias). The function can be imported via

    from mlxtend.evaluate import plot_learning_curves

Please see the implementation for the [default parameters](./mlxtend/evaluate/learning_curves.py#L7-53).

<br>
<br>


##### Examples

For more examples, please see this [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/evaluate_plot_learning_curves.ipynb)



#### Example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_learning_curves_1.png)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/evaluate_plot_learning_curves_2.png)

	from mlxtend.evaluate import plot_learning_curves
	from sklearn import datasets
	from sklearn.cross_validation import train_test_split

	# Loading some example data
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target

	X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_seed=2)

	from sklearn.tree import DecisionTreeClassifier
	import numpy as np

	clf = DecisionTreeClassifier(max_depth=1)

	plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='training_size')
	plt.show()

	plot_learning_curves(X_train, y_train, X_test, y_test, clf, kind='n_features')
	plt.show()

<br>
<br>











<a id='classifier'></a>
## Classifier

[[back to top](#overview)]

Algorithms for classification.

The `preprocessing utilities` can be imported via

	from mxtend.classifier import ...
	
<br>
<br>
<a id='perceptron'></a>
### Perceptron

[[back to top](#overview)]

Implementation of a Perceptron (single-layer artificial neural network) using the Rosenblatt Perceptron Rule [1].

[1] F. Rosenblatt. The perceptron, a perceiving and recognizing automaton Project Para. Cornell Aeronautical Laboratory, 1957.

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_perceptron_schematic.png)

For more usage examples please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/classifier_perceptron.ipynb).

A detailed explanation about the perceptron learning algorithm can be found here [Artificial Neurons and Single-Layer Neural Networks
- How Machine Learning Algorithms Work Part 1] (http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html).

<br>
<br>
##### Example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_perceptron_ros_1.png)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_perceptron_ros_2.png)

	from mlxtend.data import iris_data
	from mlxtend.evaluate import plot_decision_regions
	from mlxtend.classifier import Perceptron
	import matplotlib.pyplot as plt

	# Loading Data

	X, y = iris_data()
	X = X[:, [0, 3]] # sepal length and petal width
	X = X[0:100] # class 0 and class 1
	y = y[0:100] # class 0 and class 1

	# standardize
	X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


	# Rosenblatt Perceptron

	ppn = Perceptron(epochs=15, eta=0.01, random_seed=1)
	ppn.fit(X, y)

	plot_decision_regions(X, y, clf=ppn)
	plt.title('Perceptron - Rosenblatt Perceptron Rule')
	plt.show()

	print(ppn.w_)

	plt.plot(range(len(ppn.cost_)), ppn.cost_)
	plt.xlabel('Iterations')
	plt.ylabel('Missclassifications')
	plt.show()



<br>
<br>
##### Default Parameters

    class LogisticRegression(object):
        """Logistic regression classifier.

        Parameters
        ------------
        eta : float
          Learning rate (between 0.0 and 1.0)

        epochs : int
          Passes over the training dataset.

        learning : str (default: sgd)
          Learning rule, sgd (stochastic gradient descent)
          or gd (gradient descent).

        lambda_ : float
          Regularization parameter for L2 regularization.
          No regularization if lambda_=0.0.

        shuffle : bool (default: False)
            Shuffles training data every epoch if True to prevent circles.
        
        random_seed : int (default: None)
            Set random state for shuffling and initializing the weights.

        Attributes
        -----------
        w_ : 1d-array
          Weights after fitting.

        cost_ : list
          List of floats with sum of squared error cost (sgd or gd) for every
          epoch.

        """


<br>
<br>






<a id='adeline'></a>
### Adaline

[[back to top](#overview)]

Implementation of Adaline (Adaptive Linear Neuron; a single-layer artificial neural network) using the Widrow-Hoff delta rule. [2].

[2] B. Widrow, M. E. Hoff, et al. Adaptive switching circuits. 1960.

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_adaline_schematic.png)



For more usage examples please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/classifier_adaline.ipynb).


A detailed explanation about the Adeline learning algorithm can be found here [Artificial Neurons and Single-Layer Neural Networks
- How Machine Learning Algorithms Work Part 1] (http://sebastianraschka.com/Articles/2015_singlelayer_neurons.html).

<br>
<br>
##### Example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_adaline_sgd_1.png)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_adaline_sgd_2.png)

	from mlxtend.data import iris_data
	from mlxtend.evaluate import plot_decision_regions
	from mlxtend.classifier import Adeline
	import matplotlib.pyplot as plt

	# Loading Data

	X, y = iris_data()
	X = X[:, [0, 3]] # sepal length and petal width
	X = X[0:100] # class 0 and class 1
	y = y[0:100] # class 0 and class 1

	# standardize
	X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()


	ada = Adaline(epochs=30, eta=0.01, learning='sgd', random_seed=1)
	ada.fit(X, y)
	plot_decision_regions(X, y, clf=ada)
	plt.title('Adaline - Stochastic Gradient Descent')
	plt.show()

	plt.plot(range(len(ada.cost_)), ada.cost_, marker='o')
	plt.xlabel('Iterations')
	plt.ylabel('Cost')
	plt.show()



<br>
<br>
##### Default Parameters

    class Adaline(object):
        """ ADAptive LInear NEuron classifier.

        Parameters
        ------------
        eta : float
          Learning rate (between 0.0 and 1.0)

        epochs : int
          Passes over the training dataset.

        learning : str (default: sgd)
          Gradient decent (gd) or stochastic gradient descent (sgd)
      
        shuffle : bool (default: False)
            Shuffles training data every epoch if True to prevent circles.
        
        random_seed : int (default: None)
            Set random state for shuffling and initializing the weights.
    

        Attributes
        -----------
        w_ : 1d-array
          Weights after fitting.

        cost_ : list
          Sum of squared errors after each epoch.

        """


<br>
<br>







<br>
<br>
<a id='logistic-regression'></a>
### Logistic Regression

[[back to top](#overview)]

Implementation of Logistic Regression  with different learning rules: Gradient descent and stochastic gradient descent.

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_logistic_regression_schematic.png)

For more usage examples please see the [IPython Notebook](http://nbviewer.ipython.org/github/rasbt/mlxtend/blob/master/docs/examples/classifier_logistic_regression.ipynb).

A more detailed article about the algorithms is in preparation.

<br>
<br>
##### Example

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_logistic_regression_sgd_1.png)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/classifier_logistic_regression_sgd_2.png)

	from mlxtend.data import iris_data
	from mlxtend.evaluate import plot_decision_regions
	from mlxtend.classifier import LogisticRegression
	import matplotlib.pyplot as plt

	# Loading Data

	X, y = iris_data()
	X = X[:, [0, 3]] # sepal length and petal width
	X = X[0:100] # class 0 and class 1
	y = y[0:100] # class 0 and class 1

	# standardize
	X[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
	X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()



	lr = LogisticRegression(eta=0.01, epochs=100, learning='sgd')
	lr.fit(X, y)

	plot_decision_regions(X, y, clf=lr)
	plt.title('Logistic Regression - Stochastic Gradient Descent')
	plt.show()

	print(lr.w_)

	plt.plot(range(len(lr.cost_)), lr.cost_)
	plt.xlabel('Iterations')
	plt.ylabel('Missclassifications')
	plt.show()


<br>
<br>
##### Default Parameters

    class LogisticRegression(object):
        """Logistic regression classifier.

        Parameters
        ------------
        eta : float
          Learning rate (between 0.0 and 1.0)

        epochs : int
          Passes over the training dataset.

        learning : str (default: sgd)
          Learning rule, sgd (stochastic gradient descent)
          or gd (gradient descent).

        lambda_ : float
          Regularization parameter for L2 regularization.
          No regularization if lambda_=0.0.

        Attributes
        -----------
        w_ : 1d-array
          Weights after fitting.

        cost_ : list
          List of floats with sum of squared error cost (sgd or gd) for every
          epoch.

        """

<br>
<br>
















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

A transformer class that performs column-based mean centering on a NumPy array.

<br>
    
**Examples:**

Use the `fit` method to fit the column means of a dataset (e.g., the training dataset) to a new `MeanCenterer` object. Then, call the `transform` method on the same dataset to center it at the sample mean.

	>>> from mlxtend.preprocessing import MeanCenterer
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
<a id='array-unison-shuffling'></a>
### Array Unison Shuffling

[[back to top](#overview)]

A function that shuffles 2 or more NumPy arrays in unison.

<br>
    
**Examples:**


	>>> import numpy as np
    >>> from mlxtend.preprocessing import shuffle_arrays_unison
    >>> X1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> y1 = np.array([1, 2, 3])
    >>> print(X1)
    [[1 2 3]
    [4 5 6]
    [7 8 9]]    
    >>> print(y1)
    [1 2 3]
    >>> X2, y2 = shuffle_arrays_unison(arrays=[X1, y1], random_seed=3)
    >>> print(X2)
    [[4 5 6]
    [1 2 3]
    [7 8 9]]
    >>> print(y1)
    [2 1 3]




<br>
<br>
<br>
<br>



<a id='regression'></a>

## Regression

[[back to top](#overview)]

<br>

The `text utilities` can be imported via

	from mxtend.text import ...

<br>
<br>

<a id='plotting-linear-regression-fits'></a>
### Plotting Linear Regression Fits
[[back to top](#overview)]

`lin_regplot` is a function to plot linear regression fits. 
Uses scikit-learn's `linear_model.LinearRegression` to fit the model by default, and SciPy's `stats.pearsonr` to calculate the correlation coefficient. 

##### Default parameters:

	lin_regplot(X, y, model=LinearRegression(), corr_func=pearsonr, scattercolor='blue', fit_style='k--', legend=True, xlim='auto')

Please see the [code description](./mlxtend/regression/lin_regplot.py#L12-42) for more information.

##### Example


	from mlxtend.regression import lin_regplot
	import numpy as np

	X = np.array([4, 8, 13, 26, 31, 10, 8, 30, 18, 12, 20, 5, 28, 18, 6, 31, 12,
       12, 27, 11, 6, 14, 25, 7, 13,4, 15, 21, 15])

	y = np.array([14, 24, 22, 59, 66, 25, 18, 60, 39, 32, 53, 18, 55, 41, 28, 61, 35,
       36, 52, 23, 19, 25, 73, 16, 32, 14, 31, 43, 34])

	intercept, slope, corr_coeff = lin_regplot(X[:,np.newaxis], y,)

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/regression_lin_regplot_1.png)
	

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
- johnson an

<br>
<br>
<br>
<br>

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

##### Examples

	from mlxtend.pandas import minmax_scaling

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/pandas_scaling_minmax_scaling.png)
    

##### Default Parameters

    def minmax_scaling(df, columns, min_val=0, max_val=1):
        """ 
        Min max scaling for pandas DataFrames
        
        Parameters
        ----------
        df : pandas DataFrame object.
      
        columns : array-like, shape = [n_columns]
          Array-like with pandas DataFrame column names, e.g., ['col1', 'col2', ...]
        
        min_val : `int` or `float`, optional (default=`0`)
          minimum value after rescaling.
    
        min_val : `int` or `float`, optional (default=`1`)
          maximum value after rescaling.
      
        Returns
        ----------
    
        df_new: pandas DataFrame object. 
          Copy of the DataFrame with rescaled columns.
      
        """







	
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

<a id='installation'></a>
## Installation

[[back to top](#overview)]

You can use the following command to install `mlxtend`:  
`pip install mlxtend`  
 or    
`easy_install mlxtend`  

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

`python setup.py install`  