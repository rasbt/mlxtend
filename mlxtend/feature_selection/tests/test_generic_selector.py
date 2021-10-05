from itertools import product

import numpy as np
from sklearn.linear_model import LinearRegression
from ..generic_selector import (min_max_candidates,
                                step_candidates,
                                FeatureSelector)

have_pandas = True
try:
    import pandas as pd
except ImportError:
    have_pandas = False
    
def test_exhaustive_selector():

    n, p = 100, 5
    X = np.random.standard_normal((n, p))
    Y = np.random.standard_normal(n)

    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = min_max_candidates(X,
                                          max_features=4,
                                          fixed_features=[2,3])

    min_max_selector = FeatureSelector(LinearRegression(),
                                       initial_state,
                                       state_generator,
                                       build_submodel,
                                       check_finished)

    min_max_selector.fit(X, Y)

    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = min_max_candidates(X,
                                          max_features=4)

    min_max_selector = FeatureSelector(LinearRegression(),
                                       initial_state,
                                       state_generator,
                                       build_submodel,
                                       check_finished)

    min_max_selector.fit(X, Y)
    min_max_selector.transform(X)
    min_max_selector.fit_transform(X, Y)

    # test CV

    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = min_max_candidates(X,
                                          max_features=4)

    min_max_selector = FeatureSelector(LinearRegression(),
                                       initial_state,
                                       state_generator,
                                       build_submodel,
                                       check_finished,
                                       cv=3)
    # test CV, verbose

    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = min_max_candidates(X,
                                          max_features=4)

    for cv, verbose in product([None, 3],
                               [0,1,2]):
        min_max_selector = FeatureSelector(LinearRegression(),
                                           initial_state,
                                           state_generator,
                                           build_submodel,
                                           check_finished,
                                           cv=3,
                                           verbose=verbose)

        min_max_selector.fit(X, Y)
        print(min_max_selector.best_state_)

def test_exhaustive_categorical():

    n, p = 100, 5
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*4
    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = min_max_candidates(X,
                                          max_features=4,
                                          fixed_features=[2,3],
                                          categorical_features=categorical_features)

    min_max_selector = FeatureSelector(LinearRegression(),
                                       initial_state,
                                       state_generator,
                                       build_submodel,
                                       check_finished)

    min_max_selector.fit(X, Y)


def test_step_categorical():

    n, p = 100, 10
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*9
    (initial_state,
     state_generator,
     build_submodel,
     check_finished) = step_candidates(X,
                                       max_features=4,
                                       fixed_features=[2,3],
                                       categorical_features=categorical_features)

    step_selector = FeatureSelector(LinearRegression(),
                                    initial_state,
                                    state_generator,
                                    build_submodel,
                                    check_finished)

    step_selector.fit(X, Y)

def test_step_bigger():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    X[:,0] = np.random.choice(range(5), (n,), replace=True)
    Y = np.random.standard_normal(n)

    categorical_features = [True] + [False]*(p-1)

    for direction in ['forward', 'backward', 'both']:
        (initial_state,
         state_generator,
         build_submodel,
         check_finished) = step_candidates(X,
                                           direction=direction,
                                           max_features=p,
                                           fixed_features=[2,3],
                                           categorical_features=categorical_features)

        step_selector = FeatureSelector(LinearRegression(),
                                        initial_state,
                                        state_generator,
                                        build_submodel,
                                        check_finished)

        step_selector.fit(X, Y)
