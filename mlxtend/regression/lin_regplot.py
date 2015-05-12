from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np

def lin_regplot(X, 
             y, 
             model=LinearRegression(), 
             corr_func=pearsonr,
             scattercolor='blue', 
             fit_style='k--', 
             legend=True,
             xlim='auto'):
    """
    Function to plot a linear regression line fit.
    
    Parameters
    ----------
    X : numpy array, shape (n_samples,)
      Samples.
                
    y : numpy array, shape (n_samples,)
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
      
    Returns
    ----------
    intercept, slope, corr_coeff: float, float, float
            
    """

    if isinstance(X, list):
        X = np.asarray(X, dtype=np.float)
    if isinstance(y, list):
        y = np.asarray(y, dtype=np.float)
    if len(X.shape) == 1:
        X = X[:, np.newaxis]
    
    model.fit(X, y)

    plt.scatter(X, y, c=scattercolor)
    
    if xlim=='auto':
        x_min, x_max = X[:,0].min(), X[:,0].max()
        x_min -= 0.2*x_min
        x_max += 0.2*x_max
        
    else:
        x_min, x_max = xlim
    
    y_min = model.predict(x_min)
    y_max = model.predict(x_max)

    plt.plot([x_min, x_max], [y_min, y_max], fit_style, lw=1)

    if corr_func:
        corr_coeff, p = corr_func(X[:,0], y)
        intercept, slope = model.intercept_, model.coef_[0]

    if legend:
        leg_text = 'intercept: %.2f\nslope: %.2f' % (intercept, slope)
        if corr_func:
            leg_text += '\ncorr_coeff: %.2f' % corr_coeff
        plt.legend([leg_text], loc='best')
        
    return intercept, slope, corr_coeff