mlxtend
===========================

A library of extension and helper modules for Python's data analysis and machine learning libraries.

Those tools are intentionally not (yet) submitted to the main projects to avoid cluttering up the core libraries.


<br>
<br>

## Overview

- [scikit-learn utilities](#scikit-learn-utilities)
	- [FeatureSelector](#featureselector) 
- [Installation](#installation)
- [Changelog](./docs/CHANGELOG.txt)




<br>
<br>

## scikit-learn utilities

[[back to top](overview)]

<br>

The `scikit-learn utilities` can be imported via

	from mxtend.scikit-learn import ...

<br>
<br>
### FeatureSelector

[[back to top](overview)]

A feature selector for scikit-learn's Pipeline class that returns specified columns from a NumPy array; extremely useful in combination with scikit-learn's `Pipeline`.



Example in `Pipeline`:


	from mlxtend.sklearn import FeatureSelector
	from sklearn.pipeline import Pipeline
	from sklearn.naive_bayes import GaussianNB
	from sklearn.preprocessing import StandardScaler

	clf = Pipeline(steps=[
	    ('scaler', StandardScaler()),
    	('reduce_dim', FeatureSelector(cols=(1,3))),    # extracts column 2 and 4
    	('classifier', GaussianNB())   
    	]) 

`FeatureSelector` has a `transform` method that is used to select and return columns (features) from a NumPy array so that it can be used in the `Pipeline` like other `transformation` classes. 

	cols = ColumnExtractor(cols=(1,3)).transform(X_train)
	print('First 3 rows:\n', cols[:3,:])
	
	
	First 3 rows:
 	[[ 2.3  0.3]
 	[ 3.3  2.1]
 	[ 3.   1.2]]
        
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