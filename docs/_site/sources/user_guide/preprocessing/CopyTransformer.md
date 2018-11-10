# CopyTransformer

A simple transformer that returns a copy of the input array, for example, as part of a scikit-learn pipeline.

> from mlxtend.preprocessing import CopyTransformer

## Example 1


```python
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from mlxtend.preprocessing import CopyTransformer
import re
import numpy as np

X_train = np.array(['abc def ghi', 'this is a test',
                    'this is a test', 'this is a test'])
y_train = np.array([0, 0, 1, 1])

pipe_1 = Pipeline([
    ('vect', CountVectorizer()),
    ('to_dense', CopyTransformer()),
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


    [Parallel(n_jobs=1)]: Done  18 out of  18 | elapsed:    2.9s finished


## API


*CopyTransformer()*

Transformer that returns a copy of the input array

For usage examples, please see
[http://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/](http://rasbt.github.io/mlxtend/user_guide/preprocessing/CopyTransformer/)

### Methods

<hr>

*fit(X, y=None)*

Mock method. Does nothing.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

self

<hr>

*fit_transform(X, y=None)*

Return a copy of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_copy` : copy of the input X array.


<hr>

*get_params(deep=True)*

Get parameters for this estimator.

**Parameters**

- `deep` : boolean, optional

    If True, will return the parameters for this estimator and
    contained subobjects that are estimators.

**Returns**

- `params` : mapping of string to any

    Parameter names mapped to their values.

<hr>

*set_params(**params)*

Set the parameters of this estimator.

The method works on simple estimators as well as on nested objects
(such as pipelines). The latter have parameters of the form
``<component>__<parameter>`` so that it's possible to update each
component of a nested object.

**Returns**

self

<hr>

*transform(X, y=None)*

Return a copy of the input array.

**Parameters**

- `X` : {array-like, sparse matrix}, shape = [n_samples, n_features]

    Training vectors, where n_samples is the number of samples and
    n_features is the number of features.

- `y` : array-like, shape = [n_samples] (default: None)


**Returns**

- `X_copy` : copy of the input X array.



