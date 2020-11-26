# Sebastian Raschka 2014-2020
# mlxtend Machine Learning Library Extensions
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause
# Author: Sebastian Raschka <sebastianraschka.com>
# Author: Sebastian Raschka <sebastianraschka.com>


import seaborn as sns
import pandas as pd
import numpy as np


def scatter_hist(x, y, data):
    """ Scatter plot and histograms plot individual feature histograms along with scatter plot.

    Parameters
    ----------
    x : str or int
        DataFrame column name of the x-axis values or
        integer for the numpy ndarray column index.
    y : str
        DataFrame column name of the y-axis values or
        integer for the numpy ndarray column index
    data : Pandas DataFrame object or NumPy ndarray.
    Returns
    ---------
    plot : seaborn figure object

    """
    if isinstance(data, pd.DataFrame):
        for i in (x, y):
            assert (isinstance(i, str))

    elif isinstance(data, np.ndarray):
        for i in (x, y):
            assert (isinstance(i, int))
        x = data[:, x]
        y = data[:, y]

    else:
        raise ValueError('df must be pandas.DataFrame or numpy.ndarray object')

    plot = sns.jointplot(data=data, x=x, y=y)
    return plot
