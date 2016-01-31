# DenseTransformer

A simple transformer that converts a sparse into a dense numpy array, e.g., required for scikit-learn's `Pipeline` when, for example, `CountVectorizers` are used in combination with `RandomForest`s.

> from mlxtend.preprocessing import DenseTransformer

### Related Topics

- [Standardize](./standardize.md)
- [MeanCenterer](./mean_centerer.md)
- [Min-Max Scaling](./minmax_scaling.md)
- [DenseTransformer](./scikit-learn_dense_transformer.md)

# Examples

## Example 1


```python
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.preprocessing import DenseTransformer
import re
import numpy as np

X_train = np.array(['abc def ghi', 'this is a test',
                    'this is a test', 'this is a test'])
y_train = np.array([0, 0, 1, 1])

pipe_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('to_dense', DenseTransformer()),
    ('clf', RandomForestClassifier())
])

parameters_1 = dict(
    clf__n_estimators=[50, 100, 200],
    clf__max_features=['sqrt', 'log2', None],)

grid_search_1 = GridSearchCV(pipe_1, 
                             parameters_1, 
                             n_jobs=1, 
                             verbose=1,
                             scoring='accuracy',
                             cv=2)


print("Performing grid search...")
print("pipeline:", [name for name, _ in pipe_1.steps])
print("parameters:")
grid_search_1.fit(X_train, y_train)
print("Best score: %0.3f" % grid_search_1.best_score_)
print("Best parameters set:")
best_parameters_1 = grid_search_1.best_estimator_.get_params()
for param_name in sorted(parameters_1.keys()):
    print("\t%s: %r" % (param_name, best_parameters_1[param_name]))
```

    Performing grid search...
    pipeline: ['vect', 'to_dense', 'clf']
    parameters:
    Fitting 2 folds for each of 9 candidates, totalling 18 fits
    Best score: 0.500
    Best parameters set:
    	clf__max_features: 'sqrt'
    	clf__n_estimators: 50


    [Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:    2.4s finished


# API


*DenseTransformer(some_param=True)*

Convert a sparse matrix into a dense matrix.

### Methods

<hr>

*fit(X, y=None)*

None

<hr>

*fit_transform(X, y=None)*

None

<hr>

*get_params(deep=True)*

None

<hr>

*transform(X, y=None)*

None


