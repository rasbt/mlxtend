mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>

# Stacked Barplot

A function to conveniently plot stacked bar plots in matplotlib using pandas `DataFrame`s. 

<hr>

## Example

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
	
![](./img/matplotlib_stacked_barplot_1.png)
	
Plotting the stacked barplot. By default, the index of the `DataFrame` is used as column labels, and the `DataFrame` columns are used for the plot legend.

	from mlxtend.matplotlib import stacked_barplot

	stacked_barplot(df, rotation=45)
	
	
![](./img/matplotlib_stacked_barplot_2.png)



## Default Parameters

<pre>def stacked_barplot(df, bar_width='auto', colors='bgrcky', labels='index', rotation=90, legend=True):
    """
    Function to plot stacked barplots

    Parameters
    ----------
    df : pandas.DataFrame
      A pandas DataFrame where the index denotes the
      x-axis labels, and the columns contain the different
      measurements for each row.

    bar_width: 'auto' or float (default: 'auto')
      Parameter to set the widths of the bars. if
      'auto', the width is automatically determined by
      the number of columns in the dataset.

    colors: str (default: 'bgrcky')
      The colors of the bars.

    labels: 'index' or iterable (default: 'index')
      If 'index', the DataFrame index will be used as
      x-tick labels.

    rotation: int (default: 90)
      Parameter to rotate the x-axis labels.

    legend: bool (default: True)
      Parameter to plot the legend.

    Returns
    ----------
    None

    """</pre>