mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>

# Category Scatter

> from mlxtend.plotting import category_scatter

A function to quickly produce a scatter plot colored by categories from a pandas `DataFrame` or NumPy `ndarray` object.



<hr>
## Example

Loading an example dataset as pandas `DataFrame`:	
	
	import pandas as pd

	df = pd.read_csv('/Users/sebastian/Desktop/data.csv')
	df.head()
	
![](./img/matplotlib_categorical_scatter_1.png)
	
Plotting the data where the categories are determined by the unique values in the label column `label_col`. The `x` and `y` values are simply the column names of the DataFrame that we want to plot.

	import matplotlib.pyplot as plt
	from mlxtend.plotting import category_scatter

	category_scatter(x='x', y='y', label_col='label', data=df)
           
	plt.legend(loc='best')
	
	
![](./img/matplotlib_categorical_scatter_2.png)
	

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

![](./img/matplotlib_categorical_scatter_2.png)

<hr>
## Default Parameters

<pre>def category_scatter(x, y, label_col, data,
            markers='sxo^v',
            colors=('blue', 'green', 'red', 'purple', 'gray', 'cyan'),
            alpha=0.7, markersize=20.0):

    """
    Scatter plot to plot categories in different colors/markerstyles.
    
    Parameters
    ----------
    x : str or int
      DataFrame column name of the x-axis values or
      integer for the numpy ndarray column index.
    
    y : str
      DataFrame column name of the y-axis values or
      integer for the numpy ndarray column index
    
    data : Pandas DataFrame object or NumPy ndarray.
    
    markers : str
      Markers that are cycled through the label category.
    
    colors : tuple 
      Colors that are cycled through the label category.

    alpha : float (default: 0.7)
      Parameter to control the transparency.

    markersize : float (default : 20.0)
      Parameter to control the marker size.
      
    Returns
    ---------
    None
    
    """</pre>