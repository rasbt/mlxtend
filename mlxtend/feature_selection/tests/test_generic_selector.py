from itertools import product

import pytest

import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection.generic_selector import FeatureSelector
from mlxtend.feature_selection.strategy import exhaustive, Stepwise


have_pandas = True
try:
    import pandas as pd
except ImportError:
    have_pandas = False
    
def test_exhaustive_selector():

    n, p = 100, 5
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    strategy = exhaustive(X,
                          max_features=4,
                          fixed_features=[2,3])

    exhaustive_selector = FeatureSelector(LinearRegression(),
                                          strategy)

    exhaustive_selector.fit(X, Y)

    strategy = exhaustive(X,
                          max_features=4)

    exhaustive_selector = FeatureSelector(LinearRegression(),
                                          strategy)

    exhaustive_selector.fit(X, Y)
    exhaustive_selector.transform(X)
    exhaustive_selector.fit_transform(X, Y)

    # test CV

    strategy = exhaustive(X,
                          max_features=5)

    exhaustive_selector = FeatureSelector(LinearRegression(),
                                          strategy,
                                          cv=3)
    # test CV, verbose

    strategy = exhaustive(X,
                          max_features=3)

    for cv, verbose in product([None, 3],
                               [0,1,2]):

        exhaustive_selector = FeatureSelector(LinearRegression(),
                                              strategy,
                                              cv=cv,
                                              verbose=verbose)

        exhaustive_selector.fit(X, Y)
        print(cv, verbose, exhaustive_selector.selected_state_)

def test_exhaustive_categorical():

    n, p = 100, 5
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*4
    strategy = exhaustive(X,
                          max_features=4,
                          fixed_features=[2,3],
                          categorical_features=categorical_features)

    exhaustive_selector = FeatureSelector(LinearRegression(),
                                          strategy)

    exhaustive_selector.fit(X, Y)


def test_step_categorical():

    n, p = 100, 10
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*9
    strategy = Stepwise.first_peak(X,
                                   max_features=4,
                                   fixed_features=[2,3],
                                   initial_features=[2,3],
                                   categorical_features=categorical_features)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy)
    step_selector.fit(X, Y)

def test_step_scoring():

    n, p = 100, 10

    def AIC(estimator, X, Y):
        Yhat = estimator.predict(X)
        return 0.5 * ((Y - Yhat)**2).sum() + 2 * X.shape[1]
    
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*9
    strategy = Stepwise.first_peak(X,
                                   max_features=4,
                                   fixed_features=[2,3],
                                   initial_features=[2,3],
                                   categorical_features=categorical_features)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring='neg_mean_squared_error')
    step_selector.fit(X, Y)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring='neg_mean_squared_error',
                                    cv=None)
    step_selector.fit(X, Y)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring=AIC,
                                    cv=None)
    step_selector.fit(X, Y)
    
def test_step_bigger():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*(p-1)

    for direction in ['forward', 'backward', 'both']:
        strategy = Stepwise.first_peak(X,
                                       direction=direction,
                                       max_features=p,
                                       fixed_features=[2,3],
                                       initial_features=[2,3],
                                       categorical_features=categorical_features)

        step_selector = FeatureSelector(LinearRegression(),
                                        strategy)

        step_selector.fit(X, Y)

@pytest.mark.skipif(not have_pandas, reason='pandas unavailable')
def test_pandas1():

    n, p = 100, 5
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)
    D = pd.DataFrame(X, columns=['A', 'B', 'C', 'D', 'E'])
    D['A'] = pd.Categorical(np.random.choice(range(5), (n,), replace=True))

    for direction in ['forward', 'backward', 'both']:

        strategy = Stepwise.first_peak(D,
                                       direction=direction,
                                       max_features=p,
                                       fixed_features=['A','C'],
                                       initial_features=['A','C'])
        
        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=3)

        step_selector.fit(D, Y)
        print(step_selector.path_)

def neg_AIC(linmod, X, Y):
    """
    Negative AIC for linear regression model
    """
    n, p = X.shape
    Yhat = linmod.predict(X)
    sigma_MLE = np.linalg.norm(Y - Yhat) / np.sqrt(n)
    # adding 2 -- one for sigma^2, one for intercept fit by LinearRegression
    return -(n * np.log(2 * np.pi * sigma_MLE**2) + n + 2 * (p + 2)) 
 
def test_boston_forward():
    """Ran the following R cell
    ```{R}
    library(MASS)
    data(Boston)
    M=step(glm(medv ~ 1, data=Boston), list(upper=glm(medv ~ ., data=Boston)), direction='forward', trace=FALSE)
    print(AIC(M))
    names(coef(M)[2:12])
    ```

    Output:

    [1] 3023.726
    [1] "lstat"   "rm"      "ptratio" "dis"     "nox"     "chas"    "black"  
    [8] "zn"      "crim"    "rad"     "tax"    
    """

    try:
        import statsmodels.api as sm
    except ImportError as e:
        warnings.warn('import of statsmodels failed')
        return
    
    selected_R = ["lstat", "rm", "ptratio", "dis",
                  "nox", "chas", "black",
                  "zn", "crim",  "rad", "tax"]

    Boston = sm.datasets.get_rdataset('Boston', package='MASS').data
    D = Boston.drop('medv', axis=1)
    X = D.values
    Y = Boston['medv']

    strategy = Stepwise.first_peak(X,
                                   max_features=X.shape[1],
                                   min_features=0,
                                   direction='forward',
                                   parsimonious=False)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring=neg_AIC,
                                    cv=None)
    step_selector.fit(X, Y)

    selected_vars = [D.columns[j] for j in step_selector.selected_state_]

    Xsel = X[:,list(step_selector.selected_state_)]
    selected_model = LinearRegression().fit(Xsel, Y)
    
    assert(sorted(selected_vars) == sorted(selected_R))
    assert(np.fabs(neg_AIC(selected_model, Xsel, Y) + 3023.726) < 0.01)

def test_boston_both():
    """Ran the following R cell
    ```{R}
    %%R
    library(MASS)
    data(Boston)
    M=step(glm(medv ~ ., data=Boston), list(upper=glm(medv ~ 1, data=Boston)), direction='both', trace=FALSE)
    print(AIC(M))
    names(coef(M)[2:12])
    ```

    Output:

    [1] 3023.726
     [1] "crim"    "zn"      "chas"    "nox"     "rm"      "dis"     "rad"    
     [8] "tax"     "ptratio" "black"   "lstat"  

    """

    try:
        import statsmodels.api as sm
    except ImportError as e:
        warnings.warn('import of statsmodels failed')
        return
    
    selected_R = ["crim", "zn", "chas", "nox", "rm", "dis", "rad",
                  "tax", "ptratio", "black", "lstat"]

    Boston = sm.datasets.get_rdataset('Boston', package='MASS').data
    D = Boston.drop('medv', axis=1)
    X = D.values
    Y = Boston['medv']

    strategy = Stepwise.first_peak(X,
                                   max_features=X.shape[1],
                                   min_features=0,
                                   direction='both',
                                   parsimonious=False)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring=neg_AIC,
                                    cv=None)
    step_selector.fit(X, Y)

    selected_vars = [D.columns[j] for j in step_selector.selected_state_]

    Xsel = X[:,list(step_selector.selected_state_)]
    selected_model = LinearRegression().fit(Xsel, Y)
    
    assert(sorted(selected_vars) == sorted(selected_R))
    assert(np.fabs(neg_AIC(selected_model, Xsel, Y) + 3023.726) < 0.01)

def test_boston_back():
    """Ran the following R cell
    ```{R}
    %%R
    library(MASS)
    data(Boston)
    M=step(glm(medv ~ ., data=Boston), list(upper=glm(medv ~ 1, data=Boston)), direction='back', trace=FALSE)
    print(AIC(M))
    names(coef(M)[2:12])
    ```

    Output:

    [1] 3023.726
     [1] "crim"    "zn"      "chas"    "nox"     "rm"      "dis"     "rad"    
     [8] "tax"     "ptratio" "black"   "lstat"  

    """

    try:
        import statsmodels.api as sm
    except ImportError as e:
        warnings.warn('import of statsmodels failed')
        return
    
    selected_R = ["crim", "zn", "chas", "nox", "rm", "dis", "rad",
                  "tax", "ptratio", "black", "lstat"]

    Boston = sm.datasets.get_rdataset('Boston', package='MASS').data
    D = Boston.drop('medv', axis=1)
    X = D.values
    Y = Boston['medv']

    strategy = Stepwise.first_peak(X,
                                   max_features=X.shape[1],
                                   direction='backward',
                                   initial_features=list(range(X.shape[1])),
                                   parsimonious=False)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring=neg_AIC,
                                    cv=None)
    step_selector.fit(X, Y)

    selected_vars = [D.columns[j] for j in step_selector.selected_state_]

    Xsel = X[:,list(step_selector.selected_state_)]
    selected_model = LinearRegression().fit(Xsel, Y)
    
    assert(sorted(selected_vars) == sorted(selected_R))
    assert(np.fabs(neg_AIC(selected_model, Xsel, Y) + 3023.726) < 0.01)

def test_boston_forward_3step():
    """Ran the following R cell
    ```{R}
    library(MASS)
    data(Boston)
    M=step(glm(medv ~ 1, data=Boston), list(upper=glm(medv ~ ., data=Boston)), direction='forward', trace=FALSE, steps=3)
    print(AIC(M))
    names(coef(M)[2:4])
    ```

    Output:

    [1] 3116.097
    [1] "lstat"       "rm"          "ptratio"   

    """

    try:
        import statsmodels.api as sm
    except ImportError as e:
        warnings.warn('import of statsmodels failed')
        return
    
    selected_R = ["lstat", "rm", "ptratio"]

    Boston = sm.datasets.get_rdataset('Boston', package='MASS').data
    D = Boston.drop('medv', axis=1)
    X = D.values
    Y = Boston['medv']

    strategy = Stepwise.first_peak(X,
                                   max_features=3,
                                   min_features=3,
                                   initial_features=[],
                                   direction='forward',
                                   parsimonious=False)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy,
                                    scoring=neg_AIC,
                                    cv=None)
    step_selector.fit(X, Y)
    selected_vars = [D.columns[j] for j in step_selector.selected_state_]

    strategy2 = Stepwise.fixed_size(X,
                                    3,
                                    max_features=X.shape[1],
                                    min_features=0,
                                    initial_features=[],
                                    direction='forward')
    step_selector2 = FeatureSelector(LinearRegression(),
                                     strategy2,
                                     scoring=neg_AIC,
                                     cv=None)
    step_selector2.fit(X, Y)
    selected_vars2 = [D.columns[j] for j in step_selector2.selected_state_]
    print(selected_vars, selected_vars2)

    Xsel = X[:,list(step_selector.selected_state_)]
    selected_model = LinearRegression().fit(Xsel, Y)
    
    assert(sorted(selected_vars) == sorted(selected_R))
    assert(sorted(selected_vars2) == sorted(selected_R))
    assert(np.fabs(neg_AIC(selected_model, Xsel, Y) + 3116.097) < 0.01)
    
