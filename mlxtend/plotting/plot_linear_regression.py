# Sebastian Raschka 2014-2017
# mlxtend Machine Learning Library Extensions
#
# Function for plotting linear regression fits via scikit-learn and matplotlib.
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: BSD 3 clause


from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np


def plot_linear_regression(X, y, model=LinearRegression(),
                           corr_func='pearsonr',
                           scattercolor='blue', fit_style='k--', legend=True,
                           xlim='auto'):
    """Plot a linear regression line fit.

    Parameters
    ----------
    X : numpy array, shape = [n_samples,]
        Samples.
    y : numpy array, shape (n_samples,)
        Target values
    model: object (default: sklearn.linear_model.LinearRegression)
        Estimator object for regression. Must implement
        a .fit() and .predict() method.
    corr_func: str or function (default: 'pearsonr')
        Uses `pearsonr` from scipy.stats if corr_func='pearsonr'.
        to compute the regression slope. If not 'pearsonr', the `corr_func`,
        the `corr_func` parameter expects a function of the form
        func(<x-array>, <y-array>) as inputs, which is expected to return
        a tuple `(<correlation_coefficient>, <some_unused_value>)`.
    scattercolor: string (default: blue)
        Color of scatter plot points.
    fit_style: string (default: k--)
        Style for the line fit.
    legend: bool (default: True)
        Plots legend with corr_coeff coef.,
        fit coef., and intercept values.
    xlim: array-like (x_min, x_max) or 'auto' (default: 'auto')
        X-axis limits for the linear line fit.

    Returns
    ----------
    regression_fit : tuple
        intercept, slope, corr_coeff (float, float, float)

    """
    if isinstance(X, list):
        X = np.asarray(X, dtype=np.float)
    if isinstance(y, list):
        y = np.asarray(y, dtype=np.float)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]

    model.fit(X, y)

    plt.scatter(X, y, c=scattercolor)

    if xlim == 'auto':
        x_min, x_max = X[:, 0].min(), X[:, 0].max()
        x_min -= 0.2 * x_min
        x_max += 0.2 * x_max

    else:
        x_min, x_max = xlim

    y_min = model.predict(x_min)
    y_max = model.predict(x_max)

    plt.plot([x_min, x_max], [y_min, y_max], fit_style, lw=1)

    if corr_func == 'pearsonr':
        corr_func = pearsonr

    corr_coeff, p = corr_func(X[:, 0], y)
    intercept, slope = model.intercept_, model.coef_[0]

    if legend:
        leg_text = 'intercept: %.2f\nslope: %.2f' % (intercept, slope)
        if corr_func:
            leg_text += '\ncorrelation: %.2f' % corr_coeff
        plt.legend([leg_text], loc='best')
    regression_fit = (intercept, slope, corr_coeff)
    return regression_fit
