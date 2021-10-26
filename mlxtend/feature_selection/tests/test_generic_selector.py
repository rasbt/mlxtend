from itertools import product

import pytest

import numpy as np
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection.generic_selector import FeatureSelector
from mlxtend.feature_selection.strategy import exhaustive, step


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
                          max_features=4)

    exhaustive_selector = FeatureSelector(LinearRegression(),
                                          strategy,
                                          cv=3)
    # test CV, verbose

    strategy = exhaustive(X,
                          max_features=4)

    for cv, verbose in product([None, 3],
                               [0,1,2]):
        exhaustive_selector = FeatureSelector(LinearRegression(),
                                              strategy,
                                              cv=3,
                                              verbose=verbose)

        exhaustive_selector.fit(X, Y)
        print(exhaustive_selector.best_state_)

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
    strategy = step(X,
                    max_features=4,
                    fixed_features=[2,3],
                    categorical_features=categorical_features)

    step_selector = FeatureSelector(LinearRegression(),
                                    strategy)
    step_selector.fit(X, Y)

def test_step_bigger():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*(p-1)

    for direction in ['forward', 'backward', 'both']:
        strategy = step(X,
                        direction=direction,
                        max_features=p,
                        fixed_features=[2,3],
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

        strategy = step(D,
                        direction=direction,
                        max_features=p,
                        fixed_features=['A','C'])
        
        step_selector = FeatureSelector(LinearRegression(),
                                        strategy,
                                        cv=3)

        step_selector.fit(D, Y)
        print(step_selector.path_)
