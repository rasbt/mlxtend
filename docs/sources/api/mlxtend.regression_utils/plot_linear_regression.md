## plot_linear_regression



*plot_linear_regression(X, y, model=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False), corr_func=<function pearsonr at 0x104f200d0>, scattercolor='blue', fit_style='k--', legend=True, xlim='auto')*

Plot a linear regression line fit.

**Parameters**


- `X` : numpy array, shape = [n_samples,]

    Samples.

- `y` : numpy array, shape (n_samples,)

    Target values
    model: object (default: sklearn.linear_model.LinearRegression)
    Estimator object for regression. Must implement
    a .fit() and .predict() method.
    corr_func: function (default: scipy.stats.pearsonr)
    function to calculate the regression
    slope.
    scattercolor: string (default: blue)
    Color of scatter plot points.
    fit_style: string (default: k--)
    Style for the line fit.
    legend: bool (default: True)
    Plots legend with corr_coeff coef.,
    fit coef., and intercept values.
    xlim: array-like (x_min, x_max) or 'auto' (default: 'auto')
    X-axis limits for the linear line fit.

**Returns**


- `regression_fit` : tuple

    intercept, slope, corr_coeff (float, float, float)