# StackingRegressor

An ensemble-learning meta-regressor for stacking regression

> from mlxtend.regressor import StackingRegressor

## Overview

Stacking regression is an ensemble learning technique to combine multiple regression models via a meta-regressor. The individual regression models are trained based on the complete training set; then, the meta-regressor is fitted based on the outputs -- meta-features -- of the individual regression models in the ensemble.

![](./StackingRegressor_files/stackingregression_overview.png)

### References


- Breiman, Leo. "[Stacked regressions.](http://link.springer.com/article/10.1023/A:1018046112532#page-1)" Machine learning 24.1 (1996): 49-64.

## Example 1 - Simple Stacked Regression


```python
from mlxtend.regressor import StackingRegressor
from mlxtend.data import boston_housing_data
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import numpy as np

# Generating a sample dataset
np.random.seed(1)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(8))
```


```python
# Initializing models

lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
svr_rbf = SVR(kernel='rbf')

stregr = StackingRegressor(regressors=[svr_lin, lr, ridge], 
                           meta_regressor=svr_rbf)

# Training the stacking classifier

stregr.fit(X, y)
stregr.predict(X)

# Evaluate and visualize the fit

print("Mean Squared Error: %.4f"
      % np.mean((stregr.predict(X) - y) ** 2))
print('Variance Score: %.4f' % stregr.score(X, y))

with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X, y, c='lightgray')
    plt.plot(X, stregr.predict(X), c='darkgreen', lw=2)

plt.show()
```

    Mean Squared Error: 0.2039
    Variance Score: 0.7049



![png](StackingRegressor_files/StackingRegressor_11_1.png)



```python
stregr
```




    StackingRegressor(meta_regressor=SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False),
             regressors=[SVR(C=1.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma='auto',
      kernel='linear', max_iter=-1, shrinking=True, tol=0.001, verbose=False), LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False), Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
       normalize=False, random_state=1, solver='auto', tol=0.001)],
             verbose=0)



## Example 2 - Stacked Regression and GridSearch

To set up a parameter grid for scikit-learn's `GridSearch`, we simply provide the estimator's names in the parameter grid -- in the special case of the meta-regressor, we append the `'meta-'` prefix.


```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

# Initializing models

lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
lasso = Lasso(random_state=1)
svr_rbf = SVR(kernel='rbf')
regressors = [svr_lin, lr, ridge, lasso]
stregr = StackingRegressor(regressors=regressors, 
                           meta_regressor=svr_rbf)

params = {'lasso__alpha': [0.1, 1.0, 10.0],
          'ridge__alpha': [0.1, 1.0, 10.0],
          'svr__C': [0.1, 1.0, 10.0],
          'meta-svr__C': [0.1, 1.0, 10.0, 100.0],
          'meta-svr__gamma': [0.1, 1.0, 10.0]}

grid = GridSearchCV(estimator=stregr, 
                    param_grid=params, 
                    cv=5,
                    refit=True)
grid.fit(X, y)

for params, mean_score, scores in grid.grid_scores_:
        print("%0.3f +/- %0.2f %r"
              % (mean_score, scores.std() / 2.0, params))
```

    -9.810 +/- 6.86 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.591 +/- 6.67 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.591 +/- 6.67 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.819 +/- 6.87 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.600 +/- 6.68 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.600 +/- 6.68 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.878 +/- 6.91 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.665 +/- 6.71 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.665 +/- 6.71 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -4.839 +/- 3.98 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -3.986 +/- 3.16 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -3.986 +/- 3.16 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.875 +/- 4.01 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.005 +/- 3.17 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.005 +/- 3.17 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -5.162 +/- 4.27 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.166 +/- 3.31 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.166 +/- 3.31 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.872 +/- 3.05 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.566 +/- 3.72 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.566 +/- 3.72 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.848 +/- 3.03 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.550 +/- 3.70 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.550 +/- 3.70 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.674 +/- 2.87 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.387 +/- 3.55 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.387 +/- 3.55 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.857 +/- 4.32 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.105 +/- 3.69 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.081 +/- 3.69 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.866 +/- 4.33 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.144 +/- 3.71 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.144 +/- 3.71 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.952 +/- 4.37 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.452 +/- 3.94 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.452 +/- 3.94 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -0.240 +/- 0.18 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.083 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.083 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.251 +/- 0.19 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.086 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.086 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.270 +/- 0.20 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.107 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.107 +/- 0.12 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -1.639 +/- 1.12 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.256 +/- 1.70 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.256 +/- 1.70 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.616 +/- 1.10 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.237 +/- 1.68 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.237 +/- 1.68 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.437 +/- 0.95 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.096 +/- 1.57 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.096 +/- 1.57 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.362 +/- 0.87 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.671 +/- 0.22 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.670 +/- 0.22 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.404 +/- 0.91 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.682 +/- 0.23 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.682 +/- 0.23 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.692 +/- 1.16 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.819 +/- 0.34 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.819 +/- 0.34 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.159 +/- 1.13 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.734 +/- 0.72 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.734 +/- 0.72 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.200 +/- 1.17 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.751 +/- 0.74 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.750 +/- 0.73 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.239 +/- 1.21 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.890 +/- 0.87 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.889 +/- 0.87 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.735 +/- 0.52 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -1.247 +/- 0.81 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -1.247 +/- 0.81 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.725 +/- 0.52 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -1.212 +/- 0.79 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -1.211 +/- 0.79 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.640 +/- 0.48 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.980 +/- 0.63 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.979 +/- 0.63 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -2.669 +/- 2.59 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.038 +/- 2.95 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.037 +/- 2.95 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.671 +/- 2.60 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.957 +/- 2.87 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.952 +/- 2.87 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.660 +/- 2.59 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.997 +/- 2.93 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.999 +/- 2.93 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -1.648 +/- 1.70 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.371 +/- 1.41 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.370 +/- 1.40 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.679 +/- 1.73 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.371 +/- 1.41 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.369 +/- 1.41 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.893 +/- 1.94 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.377 +/- 1.43 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.377 +/- 1.42 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -4.113 +/- 3.15 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -13.276 +/- 9.35 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -13.287 +/- 9.36 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -3.946 +/- 3.11 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -12.797 +/- 8.93 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -12.797 +/- 8.93 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -3.551 +/- 2.90 {'lasso__alpha': 0.1, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -9.457 +/- 6.08 {'lasso__alpha': 0.1, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -9.447 +/- 6.08 {'lasso__alpha': 0.1, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -9.941 +/- 6.89 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.716 +/- 6.70 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.716 +/- 6.70 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.953 +/- 6.90 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.725 +/- 6.71 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.725 +/- 6.71 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -10.035 +/- 6.93 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.793 +/- 6.74 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.793 +/- 6.74 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -5.238 +/- 4.24 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.240 +/- 3.29 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.240 +/- 3.29 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -5.277 +/- 4.28 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.267 +/- 3.31 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.267 +/- 3.31 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -5.584 +/- 4.56 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.480 +/- 3.48 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.480 +/- 3.48 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.649 +/- 2.88 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.364 +/- 3.56 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.364 +/- 3.56 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.625 +/- 2.86 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.343 +/- 3.55 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.343 +/- 3.55 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.430 +/- 2.69 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.172 +/- 3.39 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.172 +/- 3.39 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -6.131 +/- 4.33 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.607 +/- 3.90 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.607 +/- 3.90 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -6.150 +/- 4.34 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.653 +/- 3.94 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.653 +/- 3.94 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -6.300 +/- 4.44 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.957 +/- 4.14 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.957 +/- 4.14 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -0.286 +/- 0.21 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.118 +/- 0.13 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.118 +/- 0.13 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.290 +/- 0.21 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.122 +/- 0.13 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.122 +/- 0.13 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.263 +/- 0.19 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.162 +/- 0.14 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.161 +/- 0.14 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -1.386 +/- 0.96 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.040 +/- 1.58 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.040 +/- 1.58 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.361 +/- 0.94 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.029 +/- 1.57 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.029 +/- 1.57 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.182 +/- 0.79 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.873 +/- 1.43 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.874 +/- 1.44 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.775 +/- 1.14 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.902 +/- 0.32 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.903 +/- 0.32 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.812 +/- 1.17 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.923 +/- 0.33 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.922 +/- 0.33 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -2.085 +/- 1.44 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.080 +/- 0.47 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.079 +/- 0.47 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.208 +/- 1.22 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.865 +/- 0.87 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.864 +/- 0.87 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.218 +/- 1.23 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.881 +/- 0.89 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.877 +/- 0.89 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.369 +/- 1.39 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.031 +/- 1.05 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.034 +/- 1.05 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.532 +/- 0.38 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.878 +/- 0.57 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.878 +/- 0.57 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.524 +/- 0.37 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.847 +/- 0.55 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.848 +/- 0.55 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.445 +/- 0.33 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.669 +/- 0.43 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.670 +/- 0.43 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -2.682 +/- 2.59 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.012 +/- 2.92 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.012 +/- 2.92 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.688 +/- 2.59 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.022 +/- 2.93 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.019 +/- 2.92 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.586 +/- 2.48 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.771 +/- 2.68 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.772 +/- 2.68 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -1.901 +/- 1.93 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.385 +/- 1.42 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.385 +/- 1.42 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.933 +/- 1.96 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.388 +/- 1.42 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.387 +/- 1.42 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -2.159 +/- 2.17 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.421 +/- 1.45 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.421 +/- 1.45 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -2.620 +/- 1.60 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -8.549 +/- 5.97 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -8.543 +/- 5.97 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -2.607 +/- 1.54 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -7.940 +/- 5.42 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -7.962 +/- 5.45 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -2.615 +/- 1.28 {'lasso__alpha': 1.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -5.429 +/- 3.35 {'lasso__alpha': 1.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -5.418 +/- 3.35 {'lasso__alpha': 1.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -9.941 +/- 6.89 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.716 +/- 6.70 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.716 +/- 6.70 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.953 +/- 6.90 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.725 +/- 6.71 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.725 +/- 6.71 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -10.035 +/- 6.93 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.793 +/- 6.74 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -9.793 +/- 6.74 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 0.1}
    -5.238 +/- 4.24 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.240 +/- 3.29 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.240 +/- 3.29 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -5.277 +/- 4.28 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.267 +/- 3.31 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.267 +/- 3.31 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -5.584 +/- 4.56 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.480 +/- 3.48 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.480 +/- 3.48 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 1.0}
    -4.649 +/- 2.88 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.364 +/- 3.56 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.364 +/- 3.56 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.625 +/- 2.86 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.343 +/- 3.55 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.343 +/- 3.55 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -4.430 +/- 2.69 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.172 +/- 3.39 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -5.172 +/- 3.39 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 0.1, 'meta-svr__gamma': 10.0}
    -6.131 +/- 4.33 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.607 +/- 3.90 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.607 +/- 3.90 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -6.150 +/- 4.34 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.653 +/- 3.94 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.653 +/- 3.94 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -6.300 +/- 4.44 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.957 +/- 4.14 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -5.957 +/- 4.14 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 0.1}
    -0.286 +/- 0.21 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.118 +/- 0.13 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.118 +/- 0.13 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.290 +/- 0.21 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.122 +/- 0.13 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.122 +/- 0.13 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.263 +/- 0.19 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.162 +/- 0.14 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -0.161 +/- 0.14 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 1.0}
    -1.386 +/- 0.96 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.040 +/- 1.58 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.040 +/- 1.58 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.361 +/- 0.94 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.029 +/- 1.57 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -2.029 +/- 1.57 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.182 +/- 0.79 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.873 +/- 1.43 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.874 +/- 1.44 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 1.0, 'meta-svr__gamma': 10.0}
    -1.775 +/- 1.14 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.902 +/- 0.32 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.903 +/- 0.32 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.812 +/- 1.17 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.923 +/- 0.33 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -0.922 +/- 0.33 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -2.085 +/- 1.44 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.080 +/- 0.47 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.079 +/- 0.47 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 0.1}
    -1.208 +/- 1.22 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.865 +/- 0.87 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.864 +/- 0.87 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.218 +/- 1.23 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.881 +/- 0.89 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.877 +/- 0.89 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.369 +/- 1.39 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.031 +/- 1.05 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -1.034 +/- 1.05 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 1.0}
    -0.532 +/- 0.38 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.878 +/- 0.57 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.878 +/- 0.57 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.524 +/- 0.37 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.847 +/- 0.55 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.848 +/- 0.55 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.445 +/- 0.33 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.669 +/- 0.43 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -0.670 +/- 0.43 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 10.0, 'meta-svr__gamma': 10.0}
    -2.682 +/- 2.59 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.012 +/- 2.92 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.012 +/- 2.92 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.688 +/- 2.59 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.022 +/- 2.93 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -3.019 +/- 2.92 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.586 +/- 2.48 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.771 +/- 2.68 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -2.772 +/- 2.68 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 0.1}
    -1.901 +/- 1.93 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.385 +/- 1.42 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.385 +/- 1.42 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.933 +/- 1.96 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.388 +/- 1.42 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.387 +/- 1.42 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -2.159 +/- 2.17 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.421 +/- 1.45 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -1.421 +/- 1.45 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 1.0}
    -2.620 +/- 1.60 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -8.549 +/- 5.97 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -8.543 +/- 5.97 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 0.1, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -2.607 +/- 1.54 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -7.940 +/- 5.42 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -7.962 +/- 5.45 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 1.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -2.615 +/- 1.28 {'lasso__alpha': 10.0, 'svr__C': 0.1, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -5.429 +/- 3.35 {'lasso__alpha': 10.0, 'svr__C': 1.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}
    -5.418 +/- 3.35 {'lasso__alpha': 10.0, 'svr__C': 10.0, 'ridge__alpha': 10.0, 'meta-svr__C': 100.0, 'meta-svr__gamma': 10.0}


    /Users/Sebastian/miniconda3/lib/python3.5/site-packages/sklearn/model_selection/_search.py:662: DeprecationWarning: The grid_scores_ attribute was deprecated in version 0.18 in favor of the more elaborate cv_results_ attribute. The grid_scores_ attribute will not be available from 0.20
      DeprecationWarning)



```python
# Evaluate and visualize the fit
print("Mean Squared Error: %.4f"
      % np.mean((grid.predict(X) - y) ** 2))
print('Variance Score: %.4f' % grid.score(X, y))

with plt.style.context(('seaborn-whitegrid')):
    plt.scatter(X, y, c='lightgray')
    plt.plot(X, grid.predict(X), c='darkgreen', lw=2)

plt.show()
```

    Mean Squared Error: 0.1844
    Variance Score: 0.7331



![png](StackingRegressor_files/StackingRegressor_16_1.png)


**Note**

The `StackingRegressor` also enables grid search over the `regressors` argument. However, due to the current implementation of `GridSearchCV` in scikit-learn, it is not possible to search over both, differenct classifiers and classifier parameters at the same time. For instance, while the following parameter dictionary works

    params = {'randomforestregressor__n_estimators': [1, 100],
    'regressors': [(regr1, regr1, regr1), (regr2, regr3)]}
    
it will use the instance settings of `regr1`, `regr2`, and `regr3` and not overwrite it with the `'n_estimators'` settings from `'randomforestregressor__n_estimators': [1, 100]`.

## API


*StackingRegressor(regressors, meta_regressor, verbose=0, use_features_in_secondary=False, store_train_meta_features=False, refit=True)*

A Stacking regressor for scikit-learn estimators for regression.

**Parameters**

- `regressors` : array-like, shape = [n_regressors]

    A list of regressors.
    Invoking the `fit` method on the `StackingRegressor` will fit clones
    of those original regressors that will
    be stored in the class attribute
    `self.regr_`.

- `meta_regressor` : object

    The meta-regressor to be fitted on the ensemble of
    regressors

- `verbose` : int, optional (default=0)

    Controls the verbosity of the building process.
    - `verbose=0` (default): Prints nothing
    - `verbose=1`: Prints the number & name of the regressor being fitted
    - `verbose=2`: Prints info about the parameters of the
    regressor being fitted
    - `verbose>2`: Changes `verbose` param of the underlying regressor to
    self.verbose - 2

- `use_features_in_secondary` : bool (default: False)

    If True, the meta-regressor will be trained both on
    the predictions of the original regressors and the
    original dataset.
    If False, the meta-regressor will be trained only on
    the predictions of the original regressors.

- `store_train_meta_features` : bool (default: False)

    If True, the meta-features computed from the training data
    used for fitting the
    meta-regressor stored in the `self.train_meta_features_` array,
    which can be
    accessed after calling `fit`.


**Attributes**

- `regr_` : list, shape=[n_regressors]

    Fitted regressors (clones of the original regressors)

- `meta_regr_` : estimator

    Fitted meta-regressor (clone of the original meta-estimator)

- `coef_` : array-like, shape = [n_features]

    Model coefficients of the fitted meta-estimator

- `intercept_` : float

    Intercept of the fitted meta-estimator

- `train_meta_features` : numpy array, shape = [n_samples, len(self.regressors)]

    meta-features for training data, where n_samples is the
    number of samples
    in training data and len(self.regressors) is the number of regressors.

- `refit` : bool (default: True)

    Clones the regressors for stacking regression if True (default)
    or else uses the original ones, which will be refitted on the dataset
    upon calling the `fit` method. Setting refit=False is
    recommended if you are working with estimators that are supporting
    the scikit-learn fit/predict API interface but are not compatible
    to scikit-learn's `clone` function.

**Examples**

For usage examples, please see
    [http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingRegressor/)

### Methods

<hr>

*fit(X, y, sample_weight=None)*

Learn weight coefficients from training data for each regressor.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] or [n_samples, n_targets]

    Target values.

- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights passed as sample_weights to each regressor
    in the regressors list as well as the meta_regressor.
    Raises error if some regressor does not support
    sample_weight in the fit() method.

**Returns**

- `self` : object


<hr>

*fit_transform(X, y=None, **fit_params)*

Fit to data, then transform it.

Fits transformer to X and y with optional parameters fit_params
and returns a transformed version of X.

**Parameters**

- `X` : numpy array of shape [n_samples, n_features]

    Training set.


- `y` : numpy array of shape [n_samples]

    Target values.

**Returns**

- `X_new` : numpy array of shape [n_samples, n_features_new]

    Transformed array.

<hr>

*get_params(deep=True)*

Return estimator parameter names for GridSearch support.

<hr>

*predict(X)*

Predict target values for X.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `y_target` : array-like, shape = [n_samples] or [n_samples, n_targets]

    Predicted target values.

<hr>

*predict_meta_features(X)*

Get meta-features of test-data.

**Parameters**

- `X` : numpy array, shape = [n_samples, n_features]

    Test vectors, where n_samples is the number of samples and
    n_features is the number of features.

**Returns**

- `meta-features` : numpy array, shape = [n_samples, len(self.regressors)]

    meta-features for test data, where n_samples is the number of
    samples in test data and len(self.regressors) is the number
    of regressors.

<hr>

*score(X, y, sample_weight=None)*

Returns the coefficient of determination R^2 of the prediction.

The coefficient R^2 is defined as (1 - u/v), where u is the residual
sum of squares ((y_true - y_pred) ** 2).sum() and v is the total
sum of squares ((y_true - y_true.mean()) ** 2).sum().

The best possible score is 1.0 and it can be negative (because the

model can be arbitrarily worse). A constant model that always
predicts the expected value of y, disregarding the input features,
would get a R^2 score of 0.0.

**Parameters**

- `X` : array-like, shape = (n_samples, n_features)

    Test samples.


- `y` : array-like, shape = (n_samples) or (n_samples, n_outputs)

    True values for X.


- `sample_weight` : array-like, shape = [n_samples], optional

    Sample weights.

**Returns**

- `score` : float

    R^2 of self.predict(X) wrt. y.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

### Properties

<hr>

*coef_*

None

<hr>

*intercept_*

None


