mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>

# Enrichment Plot

A function to plot step plots of cumulative counts.


<hr>
## Example

Creating an example  `DataFrame`:	
	
    import pandas as pd
    s1 = [1.1, 1.5]
    s2 = [2.1, 1.8]
    s3 = [3.1, 2.1]
    s4 = [3.9, 2.5]
    data = [s1, s2, s3, s4]
    df = pd.DataFrame(data, columns=['X1', 'X2'])
    df
	
![](./img/matplotlib_enrichment_plot_1.png)
	
Plotting the enrichment plot. The y-axis can be interpreted as "how many samples are less or equal to the corresponding x-axis label."

    from mlxtend.matplotlib import enrichment_plot
    enrichment_plot(df)
	
	
![](./img/matplotlib_enrichment_plot_2.png)
	
<hr>

## Default Parameters

<pre>
def enrichment_plot(df, colors='bgrkcy', markers=' ', linestyles='-', alpha=0.5, lw=2,
                    legend=True, where='post', grid=True, ylabel='Count',
                    xlim='auto', ylim='auto'):
    """
    Function to plot stacked barplots

    Parameters
    ----------
    df : pandas.DataFrame
      A pandas DataFrame where columns represent the different categories.

    colors: str (default: 'bgrcky')
      The colors of the bars.
      
    markers: str (default: ' ')
      Matplotlib markerstyles, e.g,
      'sov' for square,circle, and triangle markers.

    linestyles: str (default: '-')
      Matplotlib linestyles, e.g., 
      '-,--' to cycle normal and dashed lines. Note
      that the different linestyles need to be separated by commas.

    alpha: float (default: 0.5)
      Transparency level from 0.0 to 1.0.

    lw: int or float (default: 2)
      Linewidth parameter.

    legend: bool (default: True)
      Plots legend if True.

    where: {'post', 'pre', 'mid'} (default: 'post')
      Starting location of the steps.

    grid: bool (default: True)
      Plots a grid if True.

    ylabel: str (default: 'Count')
      y-axis label.

    xlim: 'auto' or array-like [min, max]
      Min and maximum position of the x-axis range.

    ylim: 'auto' or array-like [min, max]
      Min and maximum position of the y-axis range.

    Returns
    ----------
    None

    """
</pre>