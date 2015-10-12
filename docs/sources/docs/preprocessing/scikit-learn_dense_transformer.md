mlxtend  
Sebastian Raschka, last updated: 05/14/2015


<hr>

# Dense Transformer

> from mlxtend.preprocessing import DenseTransformer

A simple transformer that converts a sparse into a dense numpy array, e.g., required for scikit-learn's `Pipeline` when e.g,. `CountVectorizers` are used in combination with `RandomForest`s.

<hr>

## Example

	from sklearn.pipeline import Pipeline
	from sklearn import metrics
	from sklearn.grid_search import GridSearchCV
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.feature_extraction.text import CountVectorizer

	from mlxtend.preprocessing import DenseTransformer


	pipe_1 = Pipeline([
	    ('vect', CountVectorizer(analyzer='word',
	                      decode_error='replace',
	                      preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()), 
	                      stop_words=stopwords,) ),
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
	                           scoring=f1_scorer,
	                           cv=10)


	print("Performing grid search...")
	print("pipeline:", [name for name, _ in pipe_1.steps])
	print("parameters:")
	grid_search_1.fit(X_train, y_train)
	print("Best score: %0.3f" % grid_search_1.best_score_)
	print("Best parameters set:")
	best_parameters_1 = grid_search_1.best_estimator_.get_params()
	for param_name in sorted(parameters_1.keys()):
	    print("\t%s: %r" % (param_name, best_parameters_1[param_name]))

## Default Parameters

