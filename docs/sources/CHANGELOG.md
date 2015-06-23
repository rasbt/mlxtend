### 0.2.7

- Added 5000 labeled trainingsamples from the MNIST handwritten digits dataset to `mlxtend.data`


### 0.2.6

- Added ordinary least square regression using different solvers (gradient and stochastic gradient descent, and the closed form solution (normal equation)
- Added option for random weight initialization to logistic regression classifier and updated l2 regularization
- Added `wine` dataset to `mlxtend.data` 
- Added `invert_axes` parameter `mlxtend.matplotlib.enrichtment_plot` to optionally plot the "Count" on the x-axis
- New `verbose` parameter for `mlxtend.sklearn.EnsembleClassifier` by [Alejandro C. Bahnsen](https://github.com/albahnsen)
- Added `mlxtend.pandas.standardizing` to standardize columns in a Pandas DataFrame
- Added parameters `linestyles` and `markers` to `mlxtend.matplotlib.enrichment_plot`
- `mlxtend.regression.lin_regplot` automatically adds np.newaxis and works w. python lists
- Added tokenizers: `mlxtend.text.extract_emoticons` and `mlxtend.text.extract_words_and_emoticons`


### 0.2.5

- Added Sequential Backward Selection (mlxtend.sklearn.SBS)
- Added `X_highlight` parameter to `mlxtend.evaluate.plot_decision_regions` for highlighting test data points.
- Added mlxtend.regression.lin_regplot to plot the fitted line from linear regression.
- Added mlxtend.matplotlib.stacked_barplot to conveniently produce stacked barplots using pandas `DataFrame`s.
- Added mlxtend.matplotlib.enrichment_plot

### 0.2.4

- Added `scoring` to `mlxtend.evaluate.learning_curves` (by user pfsq)
- Fixed setup.py bug caused by the missing README.html file
- matplotlib.category_scatter for pandas DataFrames and Numpy arrays

### 0.2.3

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

### 0.2.2

- Improved robustness of EnsembleClassifier.
- Extended plot_decision_regions() functionality for plotting 1D decision boundaries.
- Function matplotlib.plot_decision_regions was reorganized  to evaluate.plot_decision_regions .
- evaluate.plot_learning_curves() function added.
- Added Rosenblatt, gradient descent, and stochastic gradient descent perceptrons.

### 0.2.1

- Added mlxtend.pandas.minmax_scaling - a function to rescale pandas DataFrame columns.
- Slight update to the EnsembleClassifier interface (additional `voting` parameter)
- Fixed EnsembleClassifier to return correct class labels if class labels are not
  integers from 0 to n.
- Added new matplotlib function to plot decision regions of classifiers.

### 0.2.0

- Improved mlxtend.text.generalize_duplcheck to remove duplicates and prevent endless looping issue.
- Added `recursive` search parameter to mlxtend.file_io.find_files.
- Added `check_ext` parameter mlxtend.file_io.find_files to search based on file extensions.
- Default parameter to ignore invisible files for mlxtend.file_io.find.
- Added `transform` and `fit_transform` to the `EnsembleClassifier`.
- Added mlxtend.file_io.find_filegroups function.

### 0.1.9

- Implemented scikit-learn EnsembleClassifier (majority voting rule) class.

### 0.1.8

- Improvements to mlxtend.text.generalize_names to handle certain Dutch last name prefixes (van, van der, de, etc.).
- Added mlxtend.text.generalize_name_duplcheck function to apply mlxtend.text.generalize_names function to a pandas DataFrame without creating duplicates.

### 0.1.7

- Added text utilities with name generalization function.
- Added  and file_io utilities.

### 0.1.6

- Added combinations and permutations estimators.

### 0.1.5

- Added `DenseTransformer` for pipelines and grid search.


### 0.1.4

- `mean_centering` function is now a Class that creates `MeanCenterer` objects
  that can be used to fit data via the `fit` method, and center data at the column
  means via the `transform` and `fit_transform` method.


### 0.1.3

- Added `preprocessing` module and `mean_centering` function.


### 0.1.2

- Added `matplotlib` utilities and `remove_borders` function.


### 0.1.1

- Simplified code for ColumnSelector.