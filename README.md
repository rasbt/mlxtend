mlxtend
===========================

A library of extension and helper modules for Python's data analysis and machine learning libraries.

Those tools are intentionally not (yet) submitted to the main projects to avoid cluttering up the core libraries.

Link to the `mlxtend` repository on GitHub: [https://github.com/rasbt/mlxtend](https://github.com/rasbt/mlxtend).

<br>
<br>

## Overview

- [preprocessing](#preprocessing)
	- [mean_centering](#mean_centering) 
- [scikit-learn utilities](#scikit-learn-utilities)
	- [ColumnSelector for custom feature selection](#columnselector-for-custom-feature-selection) 
- [matplotlib utilities](#matplotlib-utilities)
	- [remove_borders](#remove_borders) 
- [Installation](#installation)
- [Changelog](./docs/CHANGELOG.txt)


<br>
<br>
<br>
<br>

## preprocessing

[[back to top](overview)]

A collection of different functions for various data preprocessing procedures.

The `preprocessing utilities` can be imported via

	from mxtend.preprocessing import ...
	
<br>
<br>
### mean_centering

[[back to top](overview)]

	def mean_centering(X, copy=True):
    	"""
    	Function that performs column centering.
    	Keyword arguments:
        	X: NumPy array object where each attribute/variable is
           		stored in an individual column. 
           		Also accepts 1-dimensional Python list objects.
        	copy: Returns a copy of the input array if True, otherwise
              	performs operation in-place.
              
    	"""
    
**Examples:**

	>> X
	array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])
    >> mean_centering(X)
    array([[-3, -3, -3],
       [ 0,  0,  0],
       [ 3,  3,  3]])   
    

<br>

	>> X
	[1, 2, 3]
	>> mean_centering(X)
	array([-1,  0,  1])


<br>

	import matplotlib.pyplot as plt
	import numpy as np

	X = 2 * np.random.randn(100,2) + 5

	plt.scatter(X[:,0], X[:,1])
	plt.grid()
	plt.title('Random Gaussian data w. mean=5, sigma=2')
	plt.show()

	Y = mean_centering(X)
	plt.scatter(Y[:,0], Y[:,1])
	plt.grid()
	plt.title('Data after mean centering')
	plt.show()

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/mean_centering_3.png)

<br>
<br>
<br>
<br>

## scikit-learn utilities

[[back to top](overview)]

<br>

The `scikit-learn utilities` can be imported via

	from mxtend.scikit-learn import ...

<br>
<br>
### ColumnSelector for custom feature selection

[[back to top](overview)]

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
<br>
<br>

## matplotlib utilities

[[back to top](overview)]

<br>

The `matplotlib utilities` can be imported via

	from mxtend.matplotlib import ...

<br>
<br>
### remove_borders

[[back to top](overview)]

A function to remove borders from `matplotlib` plots.

	def remove_borders(axes, left=False, bottom=False, right=True, top=True):
    	""" 
    	A function to remove chartchunk from matplotlib plots, such as axes
        	spines, ticks, and labels.
        
        	Keyword arguments:
            	axes: An iterable containing plt.gca() or plt.subplot() objects, e.g. [plt.gca()].
            	left, bottom, right, top: Boolean to specify which plot axes to hide.
            
    	"""

**Example**

![](https://raw.githubusercontent.com/rasbt/mlxtend/master/images/remove_borders_3.png)

<br>
<br>

## Installation

[[back to top](overview)]

You can use the following command to install `mlxtend`:  
`pip install mlxtend`  
 or    
`easy_install mlxtend`  

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

`python setup.py install`  