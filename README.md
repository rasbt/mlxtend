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































<br>
<br>
<a id='stacked-barplot'></a>
##	


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

