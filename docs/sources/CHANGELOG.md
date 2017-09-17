# Release Notes

---

The CHANGELOG for the current development version is available at
[https://github.com/rasbt/mlxtend/blob/master/docs/sources/CHANGELOG.md](https://github.com/rasbt/mlxtend/blob/master/docs/sources/CHANGELOG.md).

### Version 0.8.1dev (TBD)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.8.1.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.8.1.tar.gz)

##### New Features

- Added `evaluate.permutation_test`, a permutation test for hypothesis testing (or A/B testing) to test if two samples come from the same distribution. Or in other words, a procedure to test the null hypothesis that that two groups are not significantly different (e.g., a treatment and a control group). 

##### Changes

- The `'support'` column returned by `frequent_patterns.association_rules` was changed to compute the support of "antecedant union consequent", and new `antecedant support'` and `'consequent support'` column were added to avoid ambiguity. 
- Added `'leverage'` as an evaluation metric for `frequent_patterns.association_rules`

##### Bug Fixes

- 


### Version 0.8.0 (2017-09-09)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.8.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.8.0.tar.gz)

##### New Features

- Added a `mlxtend.evaluate.bootstrap` that implements the ordinary nonparametric bootstrap to bootstrap a single statistic (for example, the mean. median, R^2 of a regression fit, and so forth) [#232](https://github.com/rasbt/mlxtend/pull/232)
- `SequentialFeatureSelecor`'s `k_features` now accepts a string argument "best" or "parsimonious" for more "automated" feature selection. For instance, if "best" is provided, the feature selector will return the feature subset with the best cross-validation performance. If "parsimonious" is provided as an argument, the smallest feature subset that is within one standard error of the cross-validation performance will be selected. [#238](https://github.com/rasbt/mlxtend/pull/238)

##### Changes

- `SequentialFeatureSelector` now uses `np.nanmean` over normal mean to support scorers that may return `np.nan`  [#211](https://github.com/rasbt/mlxtend/pull/211) (via [mrkaiser](https://github.com/mrkaiser))
- The `skip_if_stuck` parameter was removed from `SequentialFeatureSelector` in favor of a more efficient implementation comparing the conditional inclusion/exclusion results (in the floating versions) to the performances of previously sampled feature sets that were cached [#237](https://github.com/rasbt/mlxtend/pull/237)
- `ExhaustiveFeatureSelector` was modified to consume substantially less memory [#195](https://github.com/rasbt/mlxtend/pull/195) (via [Adam Erickson](https://github.com/adam-erickson))

##### Bug Fixes

- Fixed a bug where the `SequentialFeatureSelector` selected a feature subset larger than then specified via the `k_features` tuple max-value [#213](https://github.com/rasbt/mlxtend/pull/213)


### Version 0.7.0 (2017-06-22)



##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.7.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.7.0.tar.gz)

##### New Features

- New [mlxtend.plotting.ecdf](http://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/) function for plotting empirical cumulative distribution functions ([#196](https://github.com/rasbt/mlxtend/pull/196)).
- New [`StackingCVRegressor`](http://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/) for stacking regressors with out-of-fold predictions to prevent overfitting ([#201](https://github.com/rasbt/mlxtend/pull/201)via [Eike Dehling](https://github.com/EikeDehling)).

##### Changes

- The TensorFlow estimator have been removed from mlxtend, since TensorFlow has now very convenient ways to build on estimators, which render those implementations obsolete.
- `plot_decision_regions` now supports plotting decision regions for more than 2 training features [#189](https://github.com/rasbt/mlxtend/pull/189), via [James Bourbeau](https://github.com/jrbourbeau)).
- Parallel execution in `mlxtend.feature_selection.SequentialFeatureSelector` and `mlxtend.feature_selection.ExhaustiveFeatureSelector` is now performed over different feature subsets instead of the different cross-validation folds to better utilize machines with multiple processors if the number of features is large ([#193](https://github.com/rasbt/mlxtend/pull/193), via [@whalebot-helmsman](https://github.com/whalebot-helmsman)).
- Raise meaningful error messages if pandas `DataFrame`s or Python lists of lists are fed into the `StackingCVClassifer` as a `fit` arguments ([198](https://github.com/rasbt/mlxtend/pull/198)).
- The `n_folds` parameter of the `StackingCVClassifier` was changed to `cv` and can now accept any kind of cross validation technique that is available from scikit-learn. For example, `StackingCVClassifier(..., cv=StratifiedKFold(n_splits=3))` or `StackingCVClassifier(..., cv=GroupKFold(n_splits=3))` ([#203](https://github.com/rasbt/mlxtend/pull/203), via [Konstantinos Paliouras](https://github.com/sque)).

##### Bug Fixes

- `SequentialFeatureSelector` now correctly accepts a `None` argument for the `scoring` parameter to infer the default scoring metric from scikit-learn classifiers and regressors ([#171](https://github.com/rasbt/mlxtend/pull/171)).
- The `plot_decision_regions` function now supports pre-existing axes objects generated via matplotlib's `plt.subplots`. ([#184](https://github.com/rasbt/mlxtend/pull/184), [see example](http://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/#example-6-working-with-existing-axes-objects-using-subplots))
- Made `math.num_combinations` and `math.num_permutations` numerically stable for large numbers of combinations and permutations ([#200](https://github.com/rasbt/mlxtend/pull/200)).


### Version 0.6.0 (2017-03-18)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.6.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.6.0.tar.gz)

##### New Features

- An `association_rules` function is implemented that allows to generate rules based on a list of frequent itemsets (via [Joshua Goerner](https://github.com/JoshuaGoerner)).

##### Changes

- Adds a black `edgecolor` to plots via `plotting.plot_decision_regions` to make markers more distinguishable from the background in `matplotlib>=2.0`.
- The `association` submodule was renamed to `frequent_patterns`.

##### Bug Fixes

- The `DataFrame` index of `apriori` results are now unique and ordered.
- Fixed typos in autompg and wine datasets (via [James Bourbeau](https://github.com/jrbourbeau)).


### Version 0.5.1 (2017-02-14)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.5.1.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.5.1.tar.gz)

##### New Features

- The `EnsembleVoteClassifier` has a new `refit` attribute that prevents refitting classifiers if `refit=False` to save computational time.
- Added a new `lift_score` function in `evaluate` to compute lift score (via [Batuhan Bardak](https://github.com/bbardakk)).
- `StackingClassifier` and `StackingRegressor` support multivariate targets if the underlying models do (via [kernc](https://github.com/kernc)).
- `StackingClassifier` has a new `use_features_in_secondary` attribute like `StackingCVClassifier`.

##### Changes

- Changed default verbosity level in `SequentialFeatureSelector` to 0
- The `EnsembleVoteClassifier` now raises a `NotFittedError` if the estimator wasn't `fit` before calling `predict`. (via [Anton Loss](https://github.com/avloss))
- Added new TensorFlow variable initialization syntax to guarantee compatibility with TensorFlow 1.0

##### Bug Fixes

- Fixed wrong default value for `k_features` in `SequentialFeatureSelector`
- Cast selected feature subsets in the `SequentialFeautureSelector` as sets to prevent the iterator from getting stuck if the `k_idx` are different permutations of the same combination (via [Zac Wellmer](https://github.com/zacwellmer)).
- Fixed an issue with learning curves that caused the performance metrics to be reversed (via [ipashchenko](https://github.com/ipashchenko))
- Fixed a bug that could occur in the `SequentialFeatureSelector` if there are similarly-well performing subsets in the floating variants (via [Zac Wellmer](https://github.com/zacwellmer)).



### Version 0.5.0 (2016-11-09)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.5.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.5.0.tar.gz)

##### New Features

- New `ExhaustiveFeatureSelector` estimator in `mlxtend.feature_selection` for evaluating all feature combinations in a specified range
- The `StackingClassifier` has a new parameter `average_probas` that is set to `True` by default to maintain the current behavior. A deprecation warning was added though, and it will default to `False` in future releases (0.6.0); `average_probas=False` will result in stacking of the level-1 predicted probabilities rather than averaging these.
- New `StackingCVClassifier` estimator in 'mlxtend.classifier' for implementing a stacking ensemble that uses cross-validation techniques for training the meta-estimator to avoid overfitting ([Reiichiro Nakano](https://github.com/reiinakano))
- New `OnehotTransactions` encoder class added to the `preprocessing` submodule for transforming transaction data into a one-hot encoded array
- The `SequentialFeatureSelector` estimator in `mlxtend.feature_selection` now is safely stoppable mid-process by control+c, and deprecated print_progress in favor of a more tunable verbose parameter ([Will McGinnis](https://github.com/wdm0006))
- New `apriori` function in `association` to extract frequent itemsets from transaction data for association rule mining
- New `checkerboard_plot` function in `plotting` to plot checkerboard tables / heat maps
- New `mcnemar_table` and `mcnemar` functions in `evaluate` to compute 2x2 contingency tables and McNemar's test

##### Changes

- All plotting functions have been moved to `mlxtend.plotting` for compatibility reasons with continuous integration services and to make the installation of `matplotlib` optional for users of `mlxtend`'s core functionality
- Added a compatibility layer for `scikit-learn 0.18` using the new `model_selection` module  while maintaining backwards compatibility to scikit-learn 0.17.

##### Bug Fixes

- `mlxtend.plotting.plot_decision_regions` now draws decision regions correctly if more than 4 class labels are present
- Raise `AttributeError` in `plot_decision_regions` when the `X_higlight` argument is a 1D array ([chkoar](https://github.com/chkoar))



### Version 0.4.2 (2016-08-24)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.4.2.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.4.2.tar.gz)
- [PDF documentation](http://sebastianraschka.com/pdf/mlxtend-latest.pdf)

##### New Features

- Added `preprocessing.CopyTransformer`, a mock class that returns copies of
imput arrays via `transform` and `fit_transform`

##### Changes

- Added AppVeyor to CI to ensure MS Windows compatibility
- Dataset are now saved as compressed .txt or .csv files rather than being imported as Python objects
- `feature_selection.SequentialFeatureSelector` now supports the selection of `k_features` using a tuple to specify a "min-max" `k_features` range
- Added "SVD solver" option to the `PrincipalComponentAnalysis`
- Raise a `AttributeError` with "not fitted" message in `SequentialFeatureSelector` if `transform` or `get_metric_dict` are called prior to `fit`
- Use small, positive bias units in `TfMultiLayerPerceptron`'s hidden layer(s) if the activations are ReLUs in order to avoid dead neurons
- Added an optional `clone_estimator` parameter to the `SequentialFeatureSelector` that defaults to `True`, avoiding the modification of the original estimator objects
- More rigorous type and shape checks in the `evaluate.plot_decision_regions` function
- `DenseTransformer` now doesn't raise and error if the input array is *not* sparse
- API clean-up using scikit-learn's `BaseEstimator` as parent class for `feature_selection.ColumnSelector`

##### Bug Fixes

- Fixed a problem when a tuple-range was provided as argument to the `SequentialFeatureSelector`'s `k_features` parameter and the scoring metric was more negative than -1 (e.g., as in scikit-learn's MSE scoring function) (wahutch](https://github.com/wahutch))
- Fixed an `AttributeError` issue when `verbose` > 1 in `StackingClassifier`
- Fixed a bug in `classifier.SoftmaxRegression` where the mean values of the offsets were used to update the bias units rather than their sum
- Fixed rare bug in MLP `_layer_mapping` functions that caused a swap between the random number generation seed when initializing weights and biases

### Version 0.4.1 (2016-05-01)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.4.1.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.4.1.tar.gz)
- [PDF documentation](http://sebastianraschka.com/pdf/mlxtend-0.4.1.pdf)

##### New Features

- New TensorFlow estimator for Linear Regression (`tf_regressor.TfLinearRegression`)
- New k-means clustering estimator ([`cluster.Kmeans`](./user_guide/cluster/Kmeans.md))
- New TensorFlow k-means clustering estimator (`tf_cluster.Kmeans`)

##### Changes

- Due to refactoring of the estimator classes, the `init_weights` parameter of the `fit` methods was globally renamed to `init_params`
- Overall performance improvements of estimators due to code clean-up and refactoring
- Added several additional checks for correct array types and more meaningful exception messages
- Added optional `dropout` to the `tf_classifier.TfMultiLayerPerceptron` classifier for regularization
- Added an optional `decay` parameter to the `tf_classifier.TfMultiLayerPerceptron` classifier for adaptive learning via an exponential decay of the learning rate eta
- Replaced old `NeuralNetMLP` by more streamlined `MultiLayerPerceptron` (`classifier.MultiLayerPerceptron`); now also with softmax in the output layer and categorical cross-entropy loss.
- Unified `init_params` parameter for fit functions to continue training where the algorithm left off (if supported)

### Version 0.4.0 (2016-04-09)

##### New Features


- New `TfSoftmaxRegression` classifier using Tensorflow (`tf_classifier.TfSoftmaxRegression`)
- New `SoftmaxRegression` classifier (`classifier.SoftmaxRegression`)
- New `TfMultiLayerPerceptron` classifier using Tensorflow (`tf_classifier.TfMultiLayerPerceptron`)
- New `StackingRegressor` ([`regressor.StackingRegressor`](./user_guide/regressor/StackingRegressor.md))
- New `StackingClassifier` ([`classifier.StackingClassifier`](./user_guide/classifier/StackingClassifier.md))
- New function for one-hot encoding of class labels ([`preprocessing.one_hot`](./user_guide/preprocessing/one-hot_encoding.md))
- Added `GridSearch` support to the `SequentialFeatureSelector` ([`feature_selection/.SequentialFeatureSelector`](./user_guide/feature_selection/SequentialFeatureSelector.md))
- `evaluate.plot_decision_regions` improvements:
    - Function now handles class y-class labels correctly if array is of type `float`
    - Correct handling of input arguments `markers` and `colors`
    - Accept an existing `Axes` via the `ax` argument
- New `print_progress` parameter for all generalized models and multi-layer neural networks for printing time elapsed, ETA, and the current cost of the current epoch
- Minibatch learning for `classifier.LogisticRegression`, `classifier.Adaline`, and `regressor.LinearRegression` plus streamlined API
- New Principal Component Analysis class via [`mlxtend.feature_extraction.PrincipalComponentAnalysis`](./user_guide/feature_extraction/PrincipalComponentAnalysis.md)
- New RBF Kernel Principal Component Analysis class via [`mlxtend.feature_extraction.RBFKernelPCA`](./user_guide/feature_extraction/RBFKernelPCA.md)
- New Linear Discriminant Analysis class via [`mlxtend.feature_extraction.LinearDiscriminantAnalysis`](./user_guide/feature_extraction/LinearDiscriminantAnalysis.md)

##### Changes

- The `column` parameter in `mlxtend.preprocessing.standardize` now defaults to `None` to standardize all columns more conveniently

### Version 0.3.0 (2016-01-31)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.3.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.3.0.tar.gz)

##### New Features

- Added a progress bar tracker to `classifier.NeuralNetMLP`
- Added a function to score predicted vs. target class labels `evaluate.scoring`
- Added confusion matrix functions to create (`evaluate.confusion_matrix`) and plot (`evaluate.plot_confusion_matrix`) confusion matrices
- New style parameter and improved axis scaling in `mlxtend.evaluate.plot_learning_curves`
- Added `loadlocal_mnist` to `mlxtend.data` for streaming MNIST from a local byte files into numpy arrays
- New `NeuralNetMLP` parameters: `random_weights`, `shuffle_init`, `shuffle_epoch`
- New `SFS` features such as the generation of pandas `DataFrame` results tables and plotting functions (with confidence intervals, standard deviation, and standard error bars)
- Added support for regression estimators in `SFS`
- Added Boston `housing dataset`
- New `shuffle` parameter for `classifier.NeuralNetMLP`

##### Changes

- The `mlxtend.preprocessing.standardize` function now optionally returns the parameters, which are estimated from the array, for re-use. A further improvement makes the `standardize` function smarter in order to avoid zero-division errors
- Cosmetic improvements to the `evaluate.plot_decision_regions` function such as hiding plot axes
- Renaming of `classifier.EnsembleClassfier` to `classifier.EnsembleVoteClassifier`
- Improved random weight initialization in `Perceptron`, `Adaline`, `LinearRegression`, and `LogisticRegression`
- Changed `learning` parameter of `mlxtend.classifier.Adaline` to `solver` and added "normal equation" as closed-form solution solver
- Hide y-axis labels in `mlxtend.evaluate.plot_decision_regions` in 1 dimensional evaluations
- Sequential Feature Selection algorithms were unified into a single `SequentialFeatureSelector` class with parameters to enable floating selection and toggle between forward and backward selection.
- Stratified sampling of MNIST (now 500x random samples from each of the 10 digit categories)
- Renaming `mlxtend.plotting` to `mlxtend.general_plotting` in order to distinguish general plotting function from specialized utility function such as `evaluate.plot_decision_regions`

### Version 0.2.9 (2015-07-14)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.2.9.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.2.9.tar.gz)

##### New Features

- Sequential Feature Selection algorithms: SFS, SFFS, SBS, and SFBS

##### Changes

- Changed `regularization` & `lambda` parameters in `LogisticRegression` to single parameter `l2_lambda`

### Version 0.2.8 (2015-06-27)

- API changes:
    - `mlxtend.sklearn.EnsembleClassifier` -> `mlxtend.classifier.EnsembleClassifier`
    -  `mlxtend.sklearn.ColumnSelector` -> `mlxtend.feature_selection.ColumnSelector`
    -  `mlxtend.sklearn.DenseTransformer` -> `mlxtend.preprocessing.DenseTransformer`
    - `mlxtend.pandas.standardizing` ->  `mlxtend.preprocessing.standardizing`
    - `mlxtend.pandas.minmax_scaling` ->  `mlxtend.preprocessing.minmax_scaling`
    -  `mlxtend.matplotlib` -> `mlxtend.plotting`
- Added momentum learning parameter (alpha coefficient) to `mlxtend.classifier.NeuralNetMLP`.
- Added adaptive learning rate (decrease constant) to `mlxtend.classifier.NeuralNetMLP`.
- `mlxtend.pandas.minmax_scaling` became `mlxtend.preprocessing.minmax_scaling`  and also supports NumPy arrays now
- `mlxtend.pandas.standardizing` became `mlxtend.preprocessing.standardizing` and now supports both NumPy arrays and pandas DataFrames; also, now `ddof` parameters to set the degrees of freedom when calculating the standard deviation

### Version 0.2.7 (2015-06-20)

- Added multilayer perceptron (feedforward artificial neural network) classifier as `mlxtend.classifier.NeuralNetMLP`.
- Added 5000 labeled trainingsamples from the MNIST handwritten digits dataset to `mlxtend.data`


### Version 0.2.6 (2015-05-08)

- Added ordinary least square regression using different solvers (gradient and stochastic gradient descent, and the closed form solution (normal equation)
- Added option for random weight initialization to logistic regression classifier and updated l2 regularization
- Added `wine` dataset to `mlxtend.data`
- Added `invert_axes` parameter `mlxtend.matplotlib.enrichtment_plot` to optionally plot the "Count" on the x-axis
- New `verbose` parameter for `mlxtend.sklearn.EnsembleClassifier` by [Alejandro C. Bahnsen](https://github.com/albahnsen)
- Added `mlxtend.pandas.standardizing` to standardize columns in a Pandas DataFrame
- Added parameters `linestyles` and `markers` to `mlxtend.matplotlib.enrichment_plot`
- `mlxtend.regression.lin_regplot` automatically adds np.newaxis and works w. python lists
- Added tokenizers: `mlxtend.text.extract_emoticons` and `mlxtend.text.extract_words_and_emoticons`


### Version 0.2.5 (2015-04-17)

- Added Sequential Backward Selection (mlxtend.sklearn.SBS)
- Added `X_highlight` parameter to `mlxtend.evaluate.plot_decision_regions` for highlighting test data points.
- Added mlxtend.regression.lin_regplot to plot the fitted line from linear regression.
- Added mlxtend.matplotlib.stacked_barplot to conveniently produce stacked barplots using pandas `DataFrame`s.
- Added mlxtend.matplotlib.enrichment_plot

### Version 0.2.4 (2015-03-15)

- Added `scoring` to `mlxtend.evaluate.learning_curves` (by user pfsq)
- Fixed setup.py bug caused by the missing README.html file
- matplotlib.category_scatter for pandas DataFrames and Numpy arrays

### Version 0.2.3 (2015-03-11)

- Added Logistic regression
- Gradient descent and stochastic gradient descent perceptron was changed
  to Adaline (Adaptive Linear Neuron)
- Perceptron and Adaline for {0, 1} classes
- Added `mlxtend.preprocessing.shuffle_arrays_unison` function to
  shuffle one or more NumPy arrays.
- Added shuffle and random seed parameter to stochastic gradient descent classifier.
- Added `rstrip` parameter to `mlxtend.file_io.find_filegroups` to allow trimming of base names.
- Added `ignore_substring` parameter to `mlxtend.file_io.find_filegroups` and `find_files`.
- Replaced .rstrip in `mlxtend.file_io.find_filegroups` with more robust regex.
- Gridsearch support for `mlxtend.sklearn.EnsembleClassifier`

### Version 0.2.2 (2015-03-01)

- Improved robustness of EnsembleClassifier.
- Extended plot_decision_regions() functionality for plotting 1D decision boundaries.
- Function matplotlib.plot_decision_regions was reorganized  to evaluate.plot_decision_regions .
- evaluate.plot_learning_curves() function added.
- Added Rosenblatt, gradient descent, and stochastic gradient descent perceptrons.

### Version 0.2.1 (2015-01-20)

- Added mlxtend.pandas.minmax_scaling - a function to rescale pandas DataFrame columns.
- Slight update to the EnsembleClassifier interface (additional `voting` parameter)
- Fixed EnsembleClassifier to return correct class labels if class labels are not
  integers from 0 to n.
- Added new matplotlib function to plot decision regions of classifiers.

### Version 0.2.0 (2015-01-13)

- Improved mlxtend.text.generalize_duplcheck to remove duplicates and prevent endless looping issue.
- Added `recursive` search parameter to mlxtend.file_io.find_files.
- Added `check_ext` parameter mlxtend.file_io.find_files to search based on file extensions.
- Default parameter to ignore invisible files for mlxtend.file_io.find.
- Added `transform` and `fit_transform` to the `EnsembleClassifier`.
- Added mlxtend.file_io.find_filegroups function.

### Version 0.1.9 (2015-01-10)

- Implemented scikit-learn EnsembleClassifier (majority voting rule) class.

### Version 0.1.8 (2015-01-07)

- Improvements to mlxtend.text.generalize_names to handle certain Dutch last name prefixes (van, van der, de, etc.).
- Added mlxtend.text.generalize_name_duplcheck function to apply mlxtend.text.generalize_names function to a pandas DataFrame without creating duplicates.

### Version 0.1.7 (2015-01-07)

- Added text utilities with name generalization function.
- Added  and file_io utilities.

### Version 0.1.6 (2015-01-04)

- Added combinations and permutations estimators.

### Version 0.1.5 (2014-12-11)

- Added `DenseTransformer` for pipelines and grid search.


### Version 0.1.4 (2014-08-20)

- `mean_centering` function is now a Class that creates `MeanCenterer` objects
  that can be used to fit data via the `fit` method, and center data at the column
  means via the `transform` and `fit_transform` method.


### Version 0.1.3 (2014-08-19)

- Added `preprocessing` module and `mean_centering` function.


### Version 0.1.2 (2014-08-19)

- Added `matplotlib` utilities and `remove_borders` function.


### Version 0.1.1 (2014-08-13)

- Simplified code for ColumnSelector.
