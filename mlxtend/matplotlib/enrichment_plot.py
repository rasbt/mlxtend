import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import cycle

def enrichment_plot(df, colors='bgrkcy', alpha=0.5, lw=2,
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
    if isinstance(df, pd.Series):
        df_temp = pd.DataFrame(df)
    else:
        df_temp = df

    color_gen = cycle(colors)
    r = range(1, len(df_temp.index)+1)
    labels = df_temp.columns

    for lab in labels:
        plt.step(sorted(df_temp[lab]), r, where=where, label=lab, color=next(color_gen), alpha=alpha, lw=lw)

    if ylim == 'auto':
        plt.ylim([np.min(r)-1, np.max(r)+1])
    else:
        plt.ylim(ylim)

    if xlim == 'auto':
        df_min, df_max = np.min(df_temp.min()), np.max(df_temp.max())
        plt.xlim([df_min-1, df_max+1])
    else:
        plt.xlim(xlim)

    if legend:
        plt.legend(loc='best')

    if grid:
        plt.grid()

    if ylabel:
        plt.ylabel('Count')
