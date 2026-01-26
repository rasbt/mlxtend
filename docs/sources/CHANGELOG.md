# Release Notes

---

The CHANGELOG for the current development version is available at
[https://github.com/rasbt/mlxtend/blob/master/docs/sources/CHANGELOG.md](https://github.com/rasbt/mlxtend/blob/master/docs/sources/CHANGELOG.md).

---

### Version 0.25.0  (TBD)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.25.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.25.0.tar.gz)

##### Changes

- Removed explicit `multi_class="multinomial"` arguments, which are deprecated in newer versions of scikit-learn, from LogisticRegression usage in examples / notebooks / tests ([#1147](https://github.com/rasbt/mlxtend/issues/1147) via [sachinn854](https://github.com/sachinn854))

- Added multiprocessing support for apriori via the `n_jobs` parameter ([#1151](https://github.com/rasbt/mlxtend/issues/1151) via [mariam851](https://github.com/mariam851))

- Fixes an edge-case bug where decision regions plots didn't have unique colors ([#1157](https://github.com/rasbt/mlxtend/issues/1157) via [mariam851](https://github.com/mariam851))


### Version 0.24.0  (13 Dec 2025)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.24.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.24.0.tar.gz)


##### Changes

- Compatibility with latest scikit-learn (1.8.0) and pandas versions (2.3.3)
- [`mlxtend/classifier/stacking_cv_classification.py`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/classifier/stacking_cv_classification.py) and [`mlxtend/regressor/stacking_cv_regression.py`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/regressor/stacking_cv_regression.py)
      - Modified `meta_features` to ensure compatibility with *scikit-learn* versions 1.4 and above by dynamically selecting between `fit_params` and `params` in `cross_val_predict`.

---

### Version 0.23.4  (25 Nov 2024)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.23.4.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.23.4.tar.gz)

##### New Features and Enhancements

- Improved memory efficiency in association rule implementations and now optional `num_itemsets` parameter  ([#1121](https://github.com/rasbt/mlxtend/issues/1121) via [zazass8](https://github.com/zazass8))

##### Changes

- `np.float_` update to support for NumPy 2.0 ([#1119](https://github.com/rasbt/mlxtend/issues/1119) via [Bot-wxt1221](https://github.com/Bot-wxt1221))


---

### Version 0.23.3  (15 Nov 2024)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.23.3.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.23.3.tar.gz)

##### New Features and Enhancements

Files updated:
  - [`mlxtend.evaluate.time_series.plot_splits`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/evaluate/time_series.py)
    - Improved `plot_splits` for better visualization of time series splits

##### Changes

  - [`mlxtend/feature_selection/exhaustive_feature_selector.py`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/feature_selection/exhaustive_feature_selector.py)
    - np.inf update to support for NumPy 2.0

### Version 0.23.2  (5 Nov 2024)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.23.2.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.23.2.tar.gz)

##### New Features and Enhancements

-  Implement the FP-Growth and FP-Max algorithms with the possibility of missing values in the input dataset. Added a new metric Representativity for the association rules generated ([#1004](https://github.com/rasbt/mlxtend/issues/1004) via [zazass8](https://github.com/zazass8)).
  Files updated:
  - ['mlxtend.frequent_patterns.fpcommon']
  - ['mlxtend.frequent_patterns.fpgrowth'](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
  - ['mlxtend.frequent_patterns.fpmax'](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpmax/)
  - ['mlxtend/feature_selection/utilities.py'](https://github.com/rasbt/mlxtend/blob/master/mlxtend/feature_selection/utilities.py)
      - Modified `_calc_score` function to ensure compatibility with *scikit-learn* versions 1.4 and above by dynamically selecting between `fit_params` and `params` in `cross_val_score`.
  - [`mlxtend.feature_selection.SequentialFeatureSelector`](https://github.com/rasbt/mlxtend/blob/master/mlxtend/feature_selection/sequential_feature_selector.py)
    - Updated negative infinity constant to be compatible with old and new (>=2.0) `numpy` versions
  - [`mlxtend.frequent_patterns.association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
    - Implemented three new metrics: Jaccard, Certainty, and Kulczynski. ([#1096](https://github.com/rasbt/mlxtend/issues/1096))
  - Integrated scikit-learn's `set_output` method into `TransactionEncoder` ([#1087](https://github.com/rasbt/mlxtend/issues/1087) via [it176131](https://github.com/it176131))

##### Changes

- [`mlxtend.frequent_patterns.fpcommon`] Added the null_values parameter in valid_input_check signature to check in case the input also includes null values. Changes the returns statements and function signatures for setup_fptree and generated_itemsets respectively to return the disabled array created and to include it as a parameter. Added code in [`mlxtend.frequent_patterns.fpcommon`] and [`mlxtend.frequent_patterns.association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) to implement the algorithms in case null values exist when null_values is True.
- [`mlxtend.frequent_patterns.association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) Added optional parameter 'return_metrics' to only return a given list of metrics, rather than every possible metric.

- Add `n_classes_` attribute to stacking classifiers for compatibility with scikit-learn 1.3 ([#1091](https://github.com/rasbt/mlxtend/issues/1091))
- Use Scipy's instead of NumPy's decompositions in PCA for improved accuracy in edge cases ([#1080](https://github.com/rasbt/mlxtend/issues/1080) via [[fkdosilovic](https://github.com/rasbt/mlxtend/issues?q=is%3Apr+is%3Aopen+author%3Afkdosilovic)])



### Version 0.23.1 (5 Jan 2024)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.23.1.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.23.1.tar.gz)

##### Changes

- Updated dependency on distutils for python 3.12 and above ([#1072](https://github.com/rasbt/mlxtend/issues/1072) via [peanutsee](https://github.com/peanutsee))




### Version 0.23.0 (22 Sep 2023)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.23.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.23.0.tar.gz)

##### Changes

- Address NumPy deprecations to make mlxtend compatible to NumPy 1.24
- Changed the signature of the `LinearRegression` model of sklearn in the test removing the `normalize` parameter as it is deprecated. ([#1036](https://github.com/rasbt/mlxtend/issues/1036))
- Add `pyproject.toml` to support PEP 518 builds ([#1065](https://github.com/rasbt/mlxtend/issues/1065) via [jmahlik](https://github.com/jmahlik))
- Fixed installation from sdist failing ([#1065](https://github.com/rasbt/mlxtend/issues/1065) via [jmahlik](https://github.com/jmahlik))
- Converted configuration to `pyproject.toml` ([#1065](https://github.com/rasbt/mlxtend/issues/1065) via [jmahlik](https://github.com/jmahlik))
- Remove `mlxtend.image`  submodule with face recognition functions due to poor `dlib` support in modern environments.

##### New Features and Enhancements

- Document how to use `SequentialFeatureSelector` and multiclass ROC AUC.

### Version 0.22.0 (4 April 2023)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.22.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.22.0.tar.gz)

##### Changes

- When [`ExhaustiveFeatureSelector`](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/) is run with `n_jobs == 1`, joblib is now disabled, which enables more immediate (live) feedback when the `verbose` mode is enabled. ([#985](https://github.com/rasbt/mlxtend/pull/985) via [Nima Sarajpoor](https://github.com/NimaSarajpoor))
- Disabled unnecessary warning in [`EnsembleVoteClassifier`](https://rasbt.github.io/mlxtend/user_guide/classifier/EnsembleVoteClassifier/) ([#941](https://github.com/rasbt/mlxtend/issues/941))
- Fixed various documentation issues ([#849](https://github.com/rasbt/mlxtend/issues/849) and [#951](https://github.com/rasbt/mlxtend/issues/951) via [Lekshmanan Natarajan](https://github.com/zuari1993))
- Fixed "Edit on GitHub" button  ([#1024](https://github.com/rasbt/mlxtend/issues/1024))

##### New Features and Enhancements

- The [`mlxtend.frequent_patterns.association_rules`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) function has a new metric - Zhang's Metric, which measures both association and dissociation. ([#980](https://github.com/rasbt/mlxtend/pull/980))
- Internal [`mlxtend.frequent_patterns.fpmax`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/) code improvement that avoids casting a sparse DataFrame into a dense NumPy array. ([#1000](https://github.com/rasbt/mlxtend/pull/1000) via [Tim Kellogg](https://github.com/tkellogg))
- The [`plot_decision_regions`](https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/) function now has a `n_jobs` parameter to parallelize the computation. (In a particular use case, on a small dataset,  there was a 21x speed-up (449 seconds vs 21 seconds on local HPC instance of 36 cores). ([#998](https://github.com/rasbt/mlxtend/pull/998) via [Khalid ElHaj](https://github.com/Ne-oL))
- Added [`mlxtend.frequent_patterns.hmine`](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/hmine/) algorithm and documentation for mining frequent itemsets using the H-Mine algorithm. ([#1020](https://github.com/rasbt/mlxtend/pull/1020) via [Fatih Sen](https://github.com/fatihsen20))

### Version 0.21.0 (09/17/2022)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.21.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.21.0.tar.gz)



##### New Features and Enhancements

- The `mlxtend.evaluate.feature_importance_permutation` function has a new `feature_groups` argument to treat user-specified feature groups as single features, which is useful for one-hot encoded features. ([#955](https://github.com/rasbt/mlxtend/pull/955))
- The `mlxtend.feature_selection.ExhaustiveFeatureSelector` and `SequentialFeatureSelector` also gained support for `feature_groups` with a behavior similar to the one described above.  ([#957](https://github.com/rasbt/mlxtend/pull/957) and [#965](https://github.com/rasbt/mlxtend/pull/965) via [Nima Sarajpoor](https://github.com/NimaSarajpoor))

##### Changes

- The `custom_feature_names` parameter was removed from the `ExhaustiveFeatureSelector` due to redundancy and to simplify the code base. The [`ExhaustiveFeatureSelector` documentation](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/) illustrates how the same behavior and outcome can be achieved using pandas DataFrames.  ([#957](https://github.com/rasbt/mlxtend/pull/957))

##### Bug Fixes

- None



### Version 0.20.0 (05/26/2022)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.20.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.20.0.tar.gz)



##### New Features and Enhancements

- The `mlxtend.evaluate.bootstrap_point632_score` now supports `fit_params`. ([#861](https://github.com/rasbt/mlxtend/pull/861))
- The `mlxtend/plotting/decision_regions.py` function now has a `contourf_kwargs` for matplotlib to change the look of the decision boundaries if desired. ([#881](https://github.com/rasbt/mlxtend/pull/881) via [[pbloem](https://github.com/pbloem)])
- Add a `norm_colormap` parameter to `mlxtend.plotting.plot_confusion_matrix`, to allow normalizing the colormap, e.g., using `matplotlib.colors.LogNorm()` ([#895](https://github.com/rasbt/mlxtend/pull/895))
- Add new `GroupTimeSeriesSplit` class for evaluation in time series tasks with support of custom groups and additional parameters in comparison with scikit-learn's `TimeSeriesSplit`. ([#915](https://github.com/rasbt/mlxtend/pull/915) via [Dmitry Labazkin](https://github.com/labdmitriy))

##### Changes

- Due to compatibility issues with newer package versions, certain functions from six.py have been removed so that mlxtend may not work anymore with Python 2.7.
- As an internal change to speed up unit testing, unit testing is now faciliated by GitHub workflows, and Travis CI and Appveyor hooks have been removed.
- Improved axis label rotation in `mlxtend.plotting.heatmap` and `mlxtend.plotting.plot_confusion_matrix` ([#872](https://github.com/rasbt/mlxtend/pull/872))
- Fix various typos in McNemar guides.
- Raises a warning if non-bool arrays are used in the frequent pattern functions `apriori`, `fpmax`, and `fpgrowth`. ([#934](https://github.com/rasbt/mlxtend/pull/934) via [NimaSarajpoor](https://github.com/rasbt/mlxtend/issues?q=is%3Apr+is%3Aopen+author%3ANimaSarajpoor))

##### Bug Fixes

- Fix unreadable labels in `heatmap` for certain colormaps. ([#852](https://github.com/rasbt/mlxtend/pull/852))
- Fix an issue in `mlxtend.plotting.plot_confusion_matrix` when string class names are passed ([#894](https://github.com/rasbt/mlxtend/pull/894))





### Version 0.19.0 (2021-09-02)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.19.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.19.0.tar.gz)



##### New Features

- Adds a second "balanced accuracy" interpretation ("balanced") to `evaluate.accuracy_score` in addition to the existing "average" option to compute the scikit-learn-style balanced accuracy. ([#764](https://github.com/rasbt/mlxtend/pull/764))
- Adds new `scatter_hist` function to `mlxtend.plotting` for generating a scattered histogram. ([#757](https://github.com/rasbt/mlxtend/issues/757) via [Maitreyee Mhasaka](https://github.com/Maitreyee1))
- The `evaluate.permutation_test` function now accepts a `paired` argument to specify to support paired permutation/randomization tests. ([#768](https://github.com/rasbt/mlxtend/pull/768))
- The `StackingCVRegressor` now also supports multi-dimensional targets similar to `StackingRegressor` via `StackingCVRegressor(..., multi_output=True)`. ([#802](https://github.com/rasbt/mlxtend/pull/802) via [Marco Tiraboschi](ChromaticIsobar))

##### Changes

- Updates unit tests for scikit-learn 0.24.1 compatibility. ([#774](https://github.com/rasbt/mlxtend/pull/774))
- `StackingRegressor` now requires setting `StackingRegressor(..., multi_output=True)` if the target is multi-dimensional; this allows for better input validation. ([#802](https://github.com/rasbt/mlxtend/pull/802))
- Removes deprecated `res` argument from `plot_decision_regions`. ([#803](https://github.com/rasbt/mlxtend/pull/803))
- Adds a `title_fontsize` parameter to `plot_learning_curves` for controlling the title font size; also the plot style is now the matplotlib default. ([#818](https://github.com/rasbt/mlxtend/pull/818))
- Internal change using `'c': 'none'` instead of `'c': ''` in `mlxtend.plotting.plot_decision_regions`'s scatterplot highlights to stay compatible with Matplotlib 3.4 and newer. ([#822](https://github.com/rasbt/mlxtend/pull/822))
- Adds a `fontcolor_threshold` parameter to the `mlxtend.plotting.plot_confusion_matrix` function as an additional option for determining the font color cut-off manually. ([#827](https://github.com/rasbt/mlxtend/pull/827))
- The `frequent_patterns.association_rules` now raises a `ValueError` if an empty frequent itemset DataFrame is passed. ([#843](https://github.com/rasbt/mlxtend/pull/843))
- The .632 and .632+ bootstrap method implemented in the `mlxtend.evaluate.bootstrap_point632_score` function now use the whole training set for the resubstitution weighting term instead of the internal training set that is a new bootstrap sample in each round. ([#844](https://github.com/rasbt/mlxtend/pull/844))

##### Bug Fixes

- Fixes a typo in the SequentialFeatureSelector documentation ([#835](https://github.com/rasbt/mlxtend/issues/835) via [João Pedro Zanlorensi Cardoso](https://github.com/joaozanlorensi))


### Version 0.18.0 (2020-11-25)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.18.0.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.18.0.tar.gz)


##### New Features

- The `bias_variance_decomp` function now supports optional `fit_params` for the estimators that are fit on bootstrap samples. ([#748](https://github.com/rasbt/mlxtend/pull/748))
- The `bias_variance_decomp` function now supports Keras estimators. ([#725](https://github.com/rasbt/mlxtend/pull/725) via [@hanzigs](https://github.com/hanzigs))
- Adds new `mlxtend.classifier.OneRClassifier` (One Rule Classfier) class, a simple rule-based classifier that is often used as a performance baseline or simple interpretable model. ([#726](https://github.com/rasbt/mlxtend/pull/726)
- Adds new `create_counterfactual` method for creating counterfactuals to explain model predictions.  ([#740](https://github.com/rasbt/mlxtend/pull/740))


##### Changes

- `permutation_test` (`mlxtend.evaluate.permutation`) ìs corrected to give the proportion of permutations whose statistic is *at least as extreme* as the one observed. ([#721](https://github.com/rasbt/mlxtend/pull/721) via [Florian Charlier](https://github.com/DarthTrevis))
- Fixes the McNemar confusion matrix layout to match the convention (and documentation), swapping the upper left and lower right cells. ([#744](https://github.com/rasbt/mlxtend/pull/744) via [mmarius](https://github.com/mmarius))

##### Bug Fixes

- The loss in `LogisticRegression` for logging purposes didn't include the L2 penalty for the first weight in the weight vector (this is not the bias unit). However, since this loss function was only used for logging purposes, and the gradient remains correct, this does not have an effect on the main code. ([#741](https://github.com/rasbt/mlxtend/pull/741))
- Fixes a bug in `bias_variance_decomp` where when the `mse` loss was used, downcasting to integers caused imprecise results for small numbers. ([#749](https://github.com/rasbt/mlxtend/pull/749))


### Version 0.17.3 (2020-07-27)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.17.3.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.17.3.tar.gz)


##### New Features

- Add `predict_proba` kwarg to bootstrap methods, to allow bootstrapping of scoring functions that take in probability values. ([#700](https://github.com/rasbt/mlxtend/pull/700) via [Adam Li](https://github.com/adam2392))
- Add a `cell_values` parameter to `mlxtend.plotting.heatmap()` to optionally suppress cell annotations by setting `cell_values=False`. ([#703](https://github.com/rasbt/mlxtend/pull/700)

##### Changes

- Implemented both `use_clones` and `fit_base_estimators` (previously `refit` in `EnsembleVoteClassifier`) for `EnsembleVoteClassifier` and `StackingClassifier`. ([#670](https://github.com/rasbt/mlxtend/pull/670) via [Katrina Ni](https://github.com/nilichen))
- Switched to using raw strings for regex in `mlxtend.text` to prevent deprecation warning in Python 3.8 ([#688](https://github.com/rasbt/mlxtend/pull/688))
- Slice data in sequential forward selection before sending to parallel backend, reducing memory consumption.

##### Bug Fixes

- Fixes axis DeprecationWarning in matplotlib v3.1.0 and newer. ([#673](https://github.com/rasbt/mlxtend/pull/673))
- Fixes an issue with using `meshgrid` in `no_information_rate` function used by the `bootstrap_point632_score` function for the .632+ estimate. ([#688](https://github.com/rasbt/mlxtend/pull/688))
- Fixes an issue in `fpmax` that could lead to incorrect support values. ([#692](https://github.com/rasbt/mlxtend/pull/692) via [Steve Harenberg](https://github.com/harenbergsd))

### Version 0.17.2 (2020-02-24)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.17.2.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.17.2.tar.gz)


##### New Features

- -

##### Changes

- The previously deprecated `OnehotTransactions` has been removed in favor of the `TransactionEncoder.`
- Removed `SparseDataFrame` support in frequent pattern mining functions in favor of pandas >=1.0's new way for working sparse data. If you used `SparseDataFrame` formats, please see pandas' migration guide at https://pandas.pydata.org/pandas-docs/stable/user_guide/sparse.html#migrating. ([#667](https://github.com/rasbt/mlxtend/pull/667))
- The `plot_confusion_matrix.py` now also accepts a matplotlib figure and axis as input to which the confusion matrix plot can be added. ([#671](https://github.com/rasbt/mlxtend/pull/634) via [Vahid Mirjalili](https://github.com/vmirly))

##### Bug Fixes

- -




### Version 0.17.1 (2020-01-28)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.17.1.zip)

- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.17.1.tar.gz)

##### New Features

- The `SequentialFeatureSelector` now supports using pre-specified feature sets via the `fixed_features` parameter. ([#578](https://github.com/rasbt/mlxtend/pull/578))
- Adds a new `accuracy_score` function to `mlxtend.evaluate` for computing basic classifcation accuracy, per-class accuracy, and average per-class accuracy. ([#624](https://github.com/rasbt/mlxtend/pull/624) via [Deepan Das](https://github.com/deepandas11))
- `StackingClassifier` and `StackingCVClassifier`now have a `decision_function` method, which serves as a preferred choice over `predict_proba` in calculating roc_auc and average_precision scores when the meta estimator is a linear model or support vector classifier. ([#634](https://github.com/rasbt/mlxtend/pull/634) via [Qiang Gu](https://github.com/qiagu))

##### Changes

- Improve the runtime performance for the `apriori` frequent itemset generating function when `low_memory=True`. Setting `low_memory=False` (default) is still faster for small itemsets, but `low_memory=True` can be much faster for large itemsets and requires less memory.  Also, input validation for  `apriori`, ̀ fpgrowth` and `fpmax` takes a significant amount of time when input pandas DataFrame is large; this is now dramatically reduced when input contains boolean values (and not zeros/ones), which is the case when using `TransactionEncoder`. ([#619](https://github.com/rasbt/mlxtend/pull/619) via [Denis Barbier](https://github.com/dbarbier))
- Add support for newer sparse pandas DataFrame for frequent itemset algorithms. Also, input validation for  `apriori`, ̀ fpgrowth` and `fpmax` runs much faster on sparse DataFrame when input pandas DataFrame contains integer values. ([#621](https://github.com/rasbt/mlxtend/pull/621) via [Denis Barbier](https://github.com/dbarbier))
- Let `fpgrowth` and `fpmax` directly work on sparse DataFrame, they were previously converted into dense Numpy arrays. ([#622](https://github.com/rasbt/mlxtend/pull/622) via [Denis Barbier](https://github.com/dbarbier))

##### Bug Fixes
- Fixes a bug in `mlxtend.plotting.plot_pca_correlation_graph` that caused the explaind variances not summing up to 1. Also, improves the runtime performance of the correlation computation and adds a missing function argument for the explained variances (eigenvalues) if users provide their own principal components. ([#593](https://github.com/rasbt/mlxtend/issues/593) via [Gabriel Azevedo Ferreira](https://github.com/Gabriel-Azevedo-Ferreira))
- Behavior of `fpgrowth` and `apriori` consistent for edgecases such as `min_support=0`. ([#573](https://github.com/rasbt/mlxtend/pull/573) via [Steve Harenberg](https://github.com/harenbergsd))
- `fpmax` returns an empty data frame now instead of raising an error if the frequent itemset set is empty. ([#573](https://github.com/rasbt/mlxtend/pull/573) via [Steve Harenberg](https://github.com/harenbergsd))
- Fixes and issue in `mlxtend.plotting.plot_confusion_matrix`, where the font-color choice for medium-dark cells was not ideal and hard to read. [#588](https://github.com/rasbt/mlxtend/pull/588) via [sohrabtowfighi](https://github.com/sohrabtowfighi))
- The `svd` mode of `mlxtend.feature_extraction.PrincipalComponentAnalysis` now also *n-1* degrees of freedom instead of *n* d.o.f. when computing the eigenvalues to match the behavior of `eigen`. [#595](https://github.com/rasbt/mlxtend/pull/595)
- Disable input validation for `StackingCVClassifier` because it causes issues if pipelines are used as input. [#606](https://github.com/rasbt/mlxtend/pull/606)



### Version 0.17.0 (2019-07-19)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.17.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.17.0.tar.gz)

##### New Features

- Added an enhancement to the existing `iris_data()` such that both the UCI Repository version of the Iris dataset as well as the corrected, original
  version of the dataset can be loaded, which has a slight difference in two data points (consistent with Fisher's paper; this is also the same as in R). (via [#539](https://github.com/rasbt/mlxtend/pull/532) via [janismdhanbad](https://github.com/janismdhanbad))
- Added optional `groups` parameter to `SequentialFeatureSelector` and `ExhaustiveFeatureSelector` `fit()` methods for forwarding to sklearn CV ([#537](https://github.com/rasbt/mlxtend/pull/537) via [arc12](https://github.com/qiaguhttps://github.com/arc12))
- Added a new `plot_pca_correlation_graph` function to the `mlxtend.plotting` submodule for plotting a PCA correlation graph. ([#544](https://github.com/rasbt/mlxtend/pull/544) via [Gabriel-Azevedo-Ferreira](https://github.com/qiaguhttps://github.com/Gabriel-Azevedo-Ferreira))
- Added a `zoom_factor` parameter to the `mlxten.plotting.plot_decision_region` function that allows users to zoom in and out of the decision region plots. ([#545](https://github.com/rasbt/mlxtend/pull/545))
- Added a function `fpgrowth` that implements the FP-Growth algorithm for mining frequent itemsets as a drop-in replacement for the existing `apriori` algorithm. ([#550](https://github.com/rasbt/mlxtend/pull/550) via [Steve Harenberg](https://github.com/harenbergsd))
- New `heatmap` function in `mlxtend.plotting`.  ([#552](https://github.com/rasbt/mlxtend/pull/552))
- Added a function `fpmax` that implements the FP-Max algorithm for mining maximal itemsets as a drop-in replacement for the `fpgrowth` algorithm. ([#553](https://github.com/rasbt/mlxtend/pull/553) via [Steve Harenberg](https://github.com/harenbergsd))
- New `figsize` parameter for the `plot_decision_regions` function in `mlxtend.plotting`. ([#555](https://github.com/rasbt/mlxtend/pull/555) via [Mirza Hasanbasic](https://github.com/kazyka))
- New `low_memory` option for the `apriori` frequent itemset generating function. Setting `low_memory=False` (default) uses a substantially optimized version of the algorithm that is 3-6x faster than the original implementation (`low_memory=True`). ([#567](https://github.com/rasbt/mlxtend/pull/567) via [jmayse](https://github.com/jmayse))
- Added numerically stable OLS methods which uses `QR decomposition` and `Singular Value Decomposition` (SVD) methods to `LinearRegression` in `mlxtend.regressor.linear_regression`. ([#575](https://github.com/rasbt/mlxtend/pull/575) via [PuneetGrov3r](https://github.com/PuneetGrov3r))

##### Changes

- Now uses the latest joblib library under the hood for multiprocessing instead of `sklearn.externals.joblib`. ([#547](https://github.com/rasbt/mlxtend/pull/547))
- Changes to `StackingCVClassifier` and `StackingCVRegressor` such that first-level models are allowed to generate output of non-numeric type. ([#562](https://github.com/rasbt/mlxtend/pull/562))


##### Bug Fixes

- Fixed documentation of `iris_data()` under `iris.py` by adding a note about differences in the iris data in R and UCI machine learning repo.
- Make sure that if the `'svd'` mode is used in PCA, the number of eigenvalues is the same as when using `'eigen'` (append 0's zeros in that case) ([#565](https://github.com/rasbt/mlxtend/pull/565))

### Version 0.16.0 (2019-05-12)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.16.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.16.0.tar.gz)

##### New Features

- `StackingCVClassifier` and `StackingCVRegressor` now support `random_state` parameter, which, together with `shuffle`, controls the randomness in the cv splitting. ([#523](https://github.com/rasbt/mlxtend/pull/523) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- `StackingCVClassifier` and `StackingCVRegressor` now have a new `drop_last_proba` parameter. It drops the last "probability" column in the feature set since if `True`,
        because it is redundant: p(y_c) = 1 - p(y_1) + p(y_2) + ... + p(y_{c-1}). This can be useful for meta-classifiers that are sensitive to perfectly collinear features. ([#532](https://github.com/rasbt/mlxtend/pull/532))
- Other stacking estimators, including `StackingClassifier`, `StackingCVClassifier` and `StackingRegressor`, support grid search over the `regressors` and even a single base regressor. ([#522](https://github.com/rasbt/mlxtend/pull/522) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- Adds multiprocessing support to `StackingCVClassifier`. ([#522](https://github.com/rasbt/mlxtend/pull/522) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- Adds multiprocessing support to `StackingCVRegressor`. ([#512](https://github.com/rasbt/mlxtend/pull/512) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
-  Now, the `StackingCVRegressor` also enables grid search over the `regressors` and even a single base regressor. When there are level-mixed parameters, `GridSearchCV` will try to replace hyperparameters in a top-down order (see the [documentation](https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/) for examples details). ([#515](https://github.com/rasbt/mlxtend/pull/512) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- Adds a `verbose` parameter to `apriori` to show the current iteration number as well as the itemset size currently being sampled. ([#519](https://github.com/rasbt/mlxtend/pull/519)
- Adds an optional `class_name` parameter to the confusion matrix function to display class names on the axis as tick marks. ([#487](https://github.com/rasbt/mlxtend/pull/487) via [sandpiturtle](https://github.com/qiaguhttps://github.com/sandpiturtle))
- Adds a `pca.e_vals_normalized_` attribute to PCA for storing the eigenvalues also in normalized form; this is commonly referred to as variance explained ratios. [#545](https://github.com/rasbt/mlxtend/pull/545)

##### Changes

- Due to new features, restructuring, and better scikit-learn support (for `GridSearchCV`, etc.) the `StackingCVRegressor`'s meta regressor is now being accessed via `'meta_regressor__*` in the parameter grid. E.g., if a `RandomForestRegressor` as meta- egressor was previously tuned via `'randomforestregressor__n_estimators'`, this has now changed to `'meta_regressor__n_estimators'`. ([#515](https://github.com/rasbt/mlxtend/pull/512) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- The same change mentioned above is now applied to other stacking estimators, including `StackingClassifier`, `StackingCVClassifier` and `StackingRegressor`. ([#522](https://github.com/rasbt/mlxtend/pull/522) via [Qiang Gu](https://github.com/qiaguhttps://github.com/qiagu))
- Automatically performs mean centering for PCA solver 'SVD' such that using SVD is always equal to using the covariance matrix approach [#545](https://github.com/rasbt/mlxtend/pull/545)

##### Bug Fixes

- The `feature_selection.ColumnSelector` now also supports column names of type `int` (in addition to `str` names) if the input is a pandas DataFrame.  ([#500](https://github.com/rasbt/mlxtend/pull/500) via [tetrar124](https://github.com/tetrar124)
- Fix unreadable labels in `plot_confusion_matrix` for imbalanced datasets if `show_absolute=True` and `show_normed=True`. ([#504](https://github.com/rasbt/mlxtend/pull/504))
- Raises a more informative error if a `SparseDataFrame` is passed to `apriori` and the dataframe has integer column names that don't start with `0` due to current limitations of the `SparseDataFrame` implementation in pandas. ([#503](https://github.com/rasbt/mlxtend/pull/503))
- SequentialFeatureSelector now supports DataFrame as input for all operating modes (forward/backward/floating). [#506](https://github.com/rasbt/mlxtend/pull/506)
- `mlxtend.evaluate.feature_importance_permutation` now correctly accepts scoring functions with proper function signature as `metric` argument. [#528](https://github.com/rasbt/mlxtend/pull/528)

### Version 0.15.0 (2019-01-19)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.15.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.15.0.tar.gz)

##### New Features

- Adds a new transformer class to `mlxtend.image`, `EyepadAlign`, that aligns face images based on the location of the eyes. ([#466](https://github.com/rasbt/mlxtend/pull/466) by [Vahid Mirjalili](https://github.com/vmirly))
- Adds a new function, `mlxtend.evaluate.bias_variance_decomp` that decomposes the loss of a regressor or classifier into bias and variance terms. ([#470](https://github.com/rasbt/mlxtend/pull/470))
- Adds a `whitening` parameter to `PrincipalComponentAnalysis`, to optionally whiten the transformed data such that the features have unit variance. ([#475](https://github.com/rasbt/mlxtend/pull/475))

##### Changes

- Changed the default solver in `PrincipalComponentAnalysis` to `'svd'` instead of `'eigen'` to improve numerical stability. ([#474](https://github.com/rasbt/mlxtend/pull/474))
- The `mlxtend.image.extract_face_landmarks` now returns `None` if no facial landmarks were detected instead of an array of all zeros. ([#466](https://github.com/rasbt/mlxtend/pull/466))


##### Bug Fixes

- The eigenvectors maybe have not been sorted in certain edge cases if solver was `'eigen'` in `PrincipalComponentAnalysis` and `LinearDiscriminantAnalysis`. ([#477](https://github.com/rasbt/mlxtend/pull/477), [#478](https://github.com/rasbt/mlxtend/pull/478))


### Version 0.14.0 (2018-11-09)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.14.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.14.0.tar.gz)

##### New Features

- Added a `scatterplotmatrix` function to the `plotting` module. ([#437](https://github.com/rasbt/mlxtend/pull/437))
- Added `sample_weight` option to `StackingRegressor`, `StackingClassifier`, `StackingCVRegressor`, `StackingCVClassifier`, `EnsembleVoteClassifier`. ([#438](https://github.com/rasbt/mlxtend/issues/438))
- Added a `RandomHoldoutSplit` class to perform a random train/valid split without rotation in `SequentialFeatureSelector`, scikit-learn `GridSearchCV` etc. ([#442](https://github.com/rasbt/mlxtend/pull/442))
- Added a `PredefinedHoldoutSplit` class to perform a train/valid split, based on user-specified indices, without rotation in `SequentialFeatureSelector`, scikit-learn `GridSearchCV` etc. ([#443](https://github.com/rasbt/mlxtend/pull/443))
- Created a new `mlxtend.image` submodule for working on image processing-related tasks. ([#457](https://github.com/rasbt/mlxtend/pull/457))
- Added a new convenience function `extract_face_landmarks` based on `dlib` to `mlxtend.image`. ([#458](https://github.com/rasbt/mlxtend/pull/458))
- Added a `method='oob'` option to the `mlxtend.evaluate.bootstrap_point632_score` method to compute the classic out-of-bag bootstrap estimate ([#459](https://github.com/rasbt/mlxtend/pull/459))
- Added a `method='.632+'` option to the `mlxtend.evaluate.bootstrap_point632_score` method to compute the .632+ bootstrap estimate that addresses the optimism bias of the .632 bootstrap ([#459](https://github.com/rasbt/mlxtend/pull/459))
- Added a new `mlxtend.evaluate.ftest` function to perform an F-test for comparing the accuracies of two or more classification models. ([#460](https://github.com/rasbt/mlxtend/pull/460))
- Added a new `mlxtend.evaluate.combined_ftest_5x2cv` function to perform an combined 5x2cv F-Test for comparing the performance of two models. ([#461](https://github.com/rasbt/mlxtend/pull/461))
- Added a new `mlxtend.evaluate.difference_proportions` test for comparing two proportions (e.g., classifier accuracies) ([#462](https://github.com/rasbt/mlxtend/pull/462))


##### Changes

- Addressed deprecations warnings in NumPy 0.15. ([#425](https://github.com/rasbt/mlxtend/pull/425))
- Because of complications in PR ([#459](https://github.com/rasbt/mlxtend/pull/459)), Python 2.7 was now dropped; since official support for Python 2.7 by the Python Software Foundation is ending in approx. 12 months anyways, this re-focussing will hopefully free up some developer time with regard to not having to worry about backward compatibility

##### Bug Fixes

- Fixed an issue with a missing import in `mlxtend.plotting.plot_confusion_matrix`. ([#428](https://github.com/rasbt/mlxtend/pull/428))

### Version 0.13.0 (2018-07-20)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.13.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.13.0.tar.gz)

##### New Features

- A meaningful error message is now raised when a cross-validation generator is used with `SequentialFeatureSelector`. ([#377](https://github.com/rasbt/mlxtend/pull/377))
- The `SequentialFeatureSelector` now accepts custom feature names via the `fit` method for more interpretable feature subset reports. ([#379](https://github.com/rasbt/mlxtend/pull/379))
- The `SequentialFeatureSelector` is now also compatible with Pandas DataFrames and uses DataFrame column-names for more interpretable feature subset reports. ([#379](https://github.com/rasbt/mlxtend/pull/379))
- `ColumnSelector` now works with Pandas DataFrames columns. ([#378](https://github.com/rasbt/mlxtend/pull/378) by [Manuel Garrido](https://github.com/manugarri))
- The `ExhaustiveFeatureSelector` estimator in `mlxtend.feature_selection` now is safely stoppable mid-process by control+c. ([#380](https://github.com/rasbt/mlxtend/pull/380))
- Two new functions, `vectorspace_orthonormalization` and `vectorspace_dimensionality` were added to `mlxtend.math` to use the Gram-Schmidt process to convert a set of linearly independent vectors into a set of orthonormal basis vectors, and to compute the dimensionality of a vectorspace, respectively. ([#382](https://github.com/rasbt/mlxtend/pull/382))
- `mlxtend.frequent_patterns.apriori` now supports pandas `SparseDataFrame`s to generate frequent itemsets. ([#404](https://github.com/rasbt/mlxtend/pull/404) via [Daniel Morales](https://github.com/rasbt/mlxtend/pull/404))
- The `plot_confusion_matrix` function now has the ability to show normalized confusion matrix coefficients in addition to or instead of absolute confusion matrix coefficients with or without a colorbar. The text display method has been changed so that the full range of the colormap is used. The default size is also now set based on the number of classes.
- Added support for merging the meta features with the original input features in `StackingRegressor` (via `use_features_in_secondary`) like it is already supported in the other Stacking classes. ([#418](https://github.com/rasbt/mlxtend/pull/418))
- Added a `support_only` to the `association_rules` function, which allow constructing association rules (based on the support metric only) for cropped input DataFrames that don't contain a complete set of antecedent and consequent support values. ([#421](https://github.com/rasbt/mlxtend/pull/421))

##### Changes

- Itemsets generated with `apriori` are now `frozenset`s ([#393](https://github.com/rasbt/mlxtend/issues/393) by [William Laney](https://github.com/WLaney) and [#394](https://github.com/rasbt/mlxtend/issues/394))
- Now raises an error if a input DataFrame to `apriori` contains non 0, 1, True, False values. [#419](https://github.com/rasbt/mlxtend/issues/419))

##### Bug Fixes

- Allow mlxtend estimators to be cloned via scikit-learn's `clone` function. ([#374](https://github.com/rasbt/mlxtend/pull/374))
- Fixes bug to allow the correct use of `refit=False` in `StackingRegressor` and `StackingCVRegressor`  ([#384](https://github.com/rasbt/mlxtend/pull/384) and ([#385](https://github.com/rasbt/mlxtend/pull/385)) by [selay01](https://github.com/selay01))
- Allow `StackingClassifier` to work with sparse matrices when `use_features_in_secondary=True`  ([#408](https://github.com/rasbt/mlxtend/issues/408) by [Floris Hoogenbook](https://github.com/FlorisHoogenboom))
- Allow `StackingCVRegressor` to work with sparse matrices when `use_features_in_secondary=True`  ([#416](https://github.com/rasbt/mlxtend/issues/416))
- Allow `StackingCVClassifier` to work with sparse matrices when `use_features_in_secondary=True`  ([#417](https://github.com/rasbt/mlxtend/issues/417))



### Version 0.12.0 (2018-21-04)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.12.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.12.0.tar.gz)

##### New Features

-  A new `feature_importance_permuation` function to compute the feature importance in classifiers and regressors via the *permutation importance* method ([#358](https://github.com/rasbt/mlxtend/pull/358))
-  The fit method of the `ExhaustiveFeatureSelector` now optionally accepts `**fit_params` for the estimator that is used for the feature selection. ([#354](https://github.com/rasbt/mlxtend/pull/354) by Zach Griffith)
-  The fit method of the `SequentialFeatureSelector` now optionally accepts
`**fit_params` for the estimator that is used for the feature selection. ([#350](https://github.com/rasbt/mlxtend/pull/350) by Zach Griffith)


##### Changes


- Replaced `plot_decision_regions` colors by a colorblind-friendly palette and adds contour lines for decision regions. ([#348](https://github.com/rasbt/mlxtend/issues/348))
- All stacking estimators now raise `NonFittedErrors` if any method for inference is called prior to fitting the estimator. ([#353](https://github.com/rasbt/mlxtend/issues/353))
- Renamed the `refit` parameter of both the `StackingClassifier` and `StackingCVClassifier` to `use_clones` to be more explicit and less misleading. ([#368](https://github.com/rasbt/mlxtend/pull/368))


##### Bug Fixes

- Various changes in the documentation and documentation tools to fix formatting issues ([#363](https://github.com/rasbt/mlxtend/pull/363))
- Fixes a bug where the `StackingCVClassifier`'s meta features were not stored in the original order when `shuffle=True` ([#370](https://github.com/rasbt/mlxtend/pull/370))
- Many documentation improvements, including links to the User Guides in the API docs ([#371](https://github.com/rasbt/mlxtend/pull/371))



### Version 0.11.0 (2018-03-14)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.11.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.11.0.tar.gz)

##### New Features

-   New function implementing the resampled paired t-test procedure (`paired_ttest_resampled`)
    to compare the performance of two models. ([#323](https://github.com/rasbt/mlxtend/issues/323))
-   New function implementing the k-fold paired t-test procedure (`paired_ttest_kfold_cv`)
    to compare the performance of two models
    (also called k-hold-out paired t-test). ([#324](https://github.com/rasbt/mlxtend/issues/324))
-   New function implementing the 5x2cv paired t-test procedure (`paired_ttest_5x2cv`) proposed by Dieterrich (1998)
    to compare the performance of two models. ([#325](https://github.com/rasbt/mlxtend/issues/325))
- A `refit` parameter was added to stacking classes (similar to the `refit` parameter in the `EnsembleVoteClassifier`), to support classifiers and regressors that follow the scikit-learn API but are not compatible with scikit-learn's `clone` function. ([#322](https://github.com/rasbt/mlxtend/issues/322))
- The `ColumnSelector` now has a `drop_axis` argument to use it in pipelines with `CountVectorizers`. ([#333](https://github.com/rasbt/mlxtend/pull/333))

##### Changes


- Raises an informative error message if `predict` or `predict_meta_features` is called prior to calling the `fit` method in `StackingRegressor` and `StackingCVRegressor`. ([#315](https://github.com/rasbt/mlxtend/issues/315))
- The `plot_decision_regions` function now automatically determines the optimal setting based on the feature dimensions and supports anti-aliasing. The old `res`  parameter has been deprecated. ([#309](https://github.com/rasbt/mlxtend/pull/309) by [Guillaume Poirier-Morency](https://github.com/arteymix))
- Apriori code is faster due to optimization in `onehot transformation` and the amount of candidates generated by the `apriori` algorithm. ([#327](https://github.com/rasbt/mlxtend/pull/327) by [Jakub Smid](https://github.com/jaksmid))
- The `OnehotTransactions` class (which is typically often used in combination with the `apriori` function for association rule mining) is now more memory efficient as it uses boolean arrays instead of integer arrays. In addition, the `OnehotTransactions` class can be now be provided with `sparse` argument to generate sparse representations of the `onehot` matrix to further improve memory efficiency. ([#328](https://github.com/rasbt/mlxtend/pull/328) by [Jakub Smid](https://github.com/jaksmid))
- The `OneHotTransactions` has been deprecated and replaced by the `TransactionEncoder`. ([#332](https://github.com/rasbt/mlxtend/pull/332)
- The `plot_decision_regions` function now has three new parameters, `scatter_kwargs`, `contourf_kwargs`, and `scatter_highlight_kwargs`, that can be used to modify the plotting style. ([#342](https://github.com/rasbt/mlxtend/pull/342) by [James Bourbeau](https://github.com/jrbourbeau))


##### Bug Fixes

- Fixed issue when class labels were provided to the `EnsembleVoteClassifier` when `refit` was set to `false`. ([#322](https://github.com/rasbt/mlxtend/issues/322))
- Allow arrays with 16-bit and 32-bit precision in `plot_decision_regions` function. ([#337](https://github.com/rasbt/mlxtend/issues/337))
- Fixed bug that raised an indexing error if the number of items was <= 1 when computing association rules using the conviction metric. ([#340](https://github.com/rasbt/mlxtend/issues/340))



### Version 0.10.0 (2017-12-22)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.10.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.10.0.tar.gz)

##### New Features

- New `store_train_meta_features` parameter for `fit` in StackingCVRegressor. if True, train meta-features are stored in `self.train_meta_features_`.
    New `pred_meta_features` method for `StackingCVRegressor`. People can get test meta-features using this method. ([#294](https://github.com/rasbt/mlxtend/pull/294) via [takashioya](https://github.com/takashioya))
- The new `store_train_meta_features` attribute and `pred_meta_features` method for the `StackingCVRegressor` were also added to the `StackingRegressor`, `StackingClassifier`, and `StackingCVClassifier` ([#299](https://github.com/rasbt/mlxtend/pull/299) & [#300](https://github.com/rasbt/mlxtend/pull/300))
- New function (`evaluate.mcnemar_tables`) for creating multiple 2x2 contigency from model predictions arrays that can be used in multiple McNemar (post-hoc) tests or Cochran's Q or F tests, etc. ([#307](https://github.com/rasbt/mlxtend/issues/307))
- New function (`evaluate.cochrans_q`) for performing Cochran's Q test to compare the accuracy of multiple classifiers. ([#310](https://github.com/rasbt/mlxtend/issues/310))

##### Changes

- Added `requirements.txt` to `setup.py`. ([#304](https://github.com/rasbt/mlxtend/issues/304) via [Colin Carrol](https://github.com/ColCarroll))


##### Bug Fixes

- Improved numerical stability for p-values computed via the the exact McNemar test ([#306](https://github.com/rasbt/mlxtend/issues/306))
- `nose` is not required to use the library ([#302](https://github.com/rasbt/mlxtend/issues/302))

### Version 0.9.1 (2017-11-19)

##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.9.1.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.9.1.tar.gz)

##### New Features

- Added `mlxtend.evaluate.bootstrap_point632_score` to evaluate the performance of estimators using the .632 bootstrap. ([#283](https://github.com/rasbt/mlxtend/pull/283))
- New `max_len` parameter for the frequent itemset generation via the `apriori` function to allow for early stopping. ([#270](https://github.com/rasbt/mlxtend/pull/270))

##### Changes

- All feature index tuples in `SequentialFeatureSelector` or now in sorted order. ([#262](https://github.com/rasbt/mlxtend/pull/262))
- The `SequentialFeatureSelector` now runs the continuation of the floating inclusion/exclusion as described in Novovicova & Kittler (1994).
Note that this didn't cause any difference in performance on any of the test scenarios but could lead to better performance in certain edge cases.
([#262](https://github.com/rasbt/mlxtend/pull/262))
- `utils.Counter` now accepts a name variable to help distinguish between multiple counters, time precision can be set with the 'precision' kwarg and the new attribute end_time holds the time the last iteration completed. ([#278](https://github.com/rasbt/mlxtend/pull/278) via [Mathew Savage](https://github.com/matsavage))


##### Bug Fixes

- Fixed an deprecation error that occured with McNemar test when using SciPy 1.0. ([#283](https://github.com/rasbt/mlxtend/pull/283))


### Version 0.9.0 (2017-10-21)


##### Downloads

- [Source code (zip)](https://github.com/rasbt/mlxtend/archive/v0.9.0.zip)
- [Source code (tar.gz)](https://github.com/rasbt/mlxtend/archive/v0.9.0.tar.gz)

##### New Features

- Added `evaluate.permutation_test`, a permutation test for hypothesis testing (or A/B testing) to test if two samples come from the same distribution. Or in other words, a procedure to test the null hypothesis that that two groups are not significantly different (e.g., a treatment and a control group). ([#250](https://github.com/rasbt/mlxtend/pull/250))
- Added `'leverage'` and `'conviction` as evaluation metrics to the `frequent_patterns.association_rules` function. ([#246](https://github.com/rasbt/mlxtend/pull/246) & [#247](https://github.com/rasbt/mlxtend/pull/247))
- Added a `loadings_` attribute to `PrincipalComponentAnalysis` to compute the factor loadings of the features on the principal components. ([#251](https://github.com/rasbt/mlxtend/pull/251))
- Allow grid search over classifiers/regressors in ensemble and stacking estimators. ([#259](https://github.com/rasbt/mlxtend/pull/259))
- New `make_multiplexer_dataset` function that creates a dataset generated by a n-bit Boolean multiplexer for evaluating supervised learning algorithms. ([#263](https://github.com/rasbt/mlxtend/pull/263))
- Added a new `BootstrapOutOfBag` class, an implementation of the out-of-bag bootstrap to evaluate supervised learning algorithms. ([#265](https://github.com/rasbt/mlxtend/pull/265))
- The parameters for `StackingClassifier`, `StackingCVClassifier`, `StackingRegressor`, `StackingCVRegressor`, and `EnsembleVoteClassifier` can now be tuned using scikit-learn's `GridSearchCV` ([#254](https://github.com/rasbt/mlxtend/pull/254) via [James Bourbeau](https://github.com/jrbourbeau))

##### Changes

- The `'support'` column returned by `frequent_patterns.association_rules` was changed to compute the support of "antecedant union consequent", and new `antecedant support'` and `'consequent support'` column were added to avoid ambiguity. ([#245](https://github.com/rasbt/mlxtend/pull/245))
- Allow the `OnehotTransactions` to be cloned via scikit-learn's `clone` function, which is required by e.g., scikit-learn's `FeatureUnion` or `GridSearchCV` (via [Iaroslav Shcherbatyi](https://github.com/iaroslav-ai)). ([#249](https://github.com/rasbt/mlxtend/pull/249))

##### Bug Fixes

- Fix issues with `self._init_time` parameter in `_IterativeModel` subclasses. ([#256](https://github.com/rasbt/mlxtend/pull/256))
- Fix imprecision bug that occurred in `plot_ecdf` when run on Python 2.7. ([264](https://github.com/rasbt/mlxtend/pull/264))
- The vectors from SVD in `PrincipalComponentAnalysis` are now being scaled so that the eigenvalues via `solver='eigen'` and `solver='svd'` now store eigenvalues that have the same magnitudes. ([#251](https://github.com/rasbt/mlxtend/pull/251))

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

- New [mlxtend.plotting.ecdf](https://rasbt.github.io/mlxtend/user_guide/plotting/ecdf/) function for plotting empirical cumulative distribution functions ([#196](https://github.com/rasbt/mlxtend/pull/196)).
- New [`StackingCVRegressor`](https://rasbt.github.io/mlxtend/user_guide/regressor/StackingCVRegressor/) for stacking regressors with out-of-fold predictions to prevent overfitting ([#201](https://github.com/rasbt/mlxtend/pull/201)via [Eike Dehling](https://github.com/EikeDehling)).

##### Changes

- The TensorFlow estimator have been removed from mlxtend, since TensorFlow has now very convenient ways to build on estimators, which render those implementations obsolete.
- `plot_decision_regions` now supports plotting decision regions for more than 2 training features [#189](https://github.com/rasbt/mlxtend/pull/189), via [James Bourbeau](https://github.com/jrbourbeau)).
- Parallel execution in `mlxtend.feature_selection.SequentialFeatureSelector` and `mlxtend.feature_selection.ExhaustiveFeatureSelector` is now performed over different feature subsets instead of the different cross-validation folds to better utilize machines with multiple processors if the number of features is large ([#193](https://github.com/rasbt/mlxtend/pull/193), via [@whalebot-helmsman](https://github.com/whalebot-helmsman)).
- Raise meaningful error messages if pandas `DataFrame`s or Python lists of lists are fed into the `StackingCVClassifer` as a `fit` arguments ([198](https://github.com/rasbt/mlxtend/pull/198)).
- The `n_folds` parameter of the `StackingCVClassifier` was changed to `cv` and can now accept any kind of cross validation technique that is available from scikit-learn. For example, `StackingCVClassifier(..., cv=StratifiedKFold(n_splits=3))` or `StackingCVClassifier(..., cv=GroupKFold(n_splits=3))` ([#203](https://github.com/rasbt/mlxtend/pull/203), via [Konstantinos Paliouras](https://github.com/sque)).

##### Bug Fixes

- `SequentialFeatureSelector` now correctly accepts a `None` argument for the `scoring` parameter to infer the default scoring metric from scikit-learn classifiers and regressors ([#171](https://github.com/rasbt/mlxtend/pull/171)).
- The `plot_decision_regions` function now supports pre-existing axes objects generated via matplotlib's `plt.subplots`. ([#184](https://github.com/rasbt/mlxtend/pull/184), [see example](https://rasbt.github.io/mlxtend/user_guide/plotting/plot_decision_regions/#example-6-working-with-existing-axes-objects-using-subplots))
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
- The `SequentialFeatureSelector` estimator in `mlxtend.feature_selection` now is safely stoppable mid-process by control+c, and deprecated `print_progress` in favor of a more tunable `verbose` parameter ([Will McGinnis](https://github.com/wdm0006))
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
- [PDF documentation](https://sebastianraschka.com/pdf/mlxtend-latest.pdf)

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
- [PDF documentation](https://sebastianraschka.com/pdf/mlxtend-0.4.1.pdf)

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
