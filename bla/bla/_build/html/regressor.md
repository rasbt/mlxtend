<div class="document">

<div class="documentwrapper">

<div class="bodywrapper">

<div class="body" role="main">

<div id="mlxtend-regressor-package" class="section">

mlxtend.regressor package[¶](#mlxtend-regressor-package "Permalink to this headline")
=====================================================================================

<div id="submodules" class="section">

Submodules[¶](#submodules "Permalink to this headline")
-------------------------------------------------------

</div>

<div id="module-mlxtend.regressor.linear_regression" class="section">

<span id="mlxtend-regressor-linear-regression-module"></span>
mlxtend.regressor.linear\_regression module[¶](#module-mlxtend.regressor.linear_regression "Permalink to this headline")
------------------------------------------------------------------------------------------------------------------------

 *class* `mlxtend.regressor.linear_regression.`{.descclassname}`LinearRegression`{.descname}<span class="sig-paren">(</span>*solver='normal\_equation'*, *eta=0.01*, *epochs=50*, *random\_seed=None*, *shuffle=False*, *zero\_init\_weight=False*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression)[¶](#mlxtend.regressor.linear_regression.LinearRegression "Permalink to this definition")

:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Ordinary least squares linear regression.

    solver <span class="classifier-delimiter">:</span> <span class="classifier">{‘gd’, ‘sgd’, ‘normal\_equation’} (default: ‘normal\_equation’)</span>
    :   Method for solving the cost function. ‘gd’ for gradient descent,
        ‘sgd’ for stochastic gradient descent, or
        ‘normal\_equation’ (default) to solve the cost
        function analytically.

    eta <span class="classifier-delimiter">:</span> <span class="classifier">float (default: 0.1)</span>
    :   Learning rate (between 0.0 and 1.0); ignored
        if solver=’normal\_equation’.

    epochs <span class="classifier-delimiter">:</span> <span class="classifier">int (default: 50)</span>
    :   Passes over the training dataset; ignored
        if solver=’normal\_equation’.

    shuffle <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: False)</span>
    :   Shuffles training data every epoch if True to prevent circles;
        ignored if solver=’normal\_equation’.

    random\_seed <span class="classifier-delimiter">:</span> <span class="classifier">int (default: None)</span>
    :   Set random state for shuffling and initializing the weights;
        ignored if solver=’normal\_equation’.

    zero\_init\_weight <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: False)</span>
    :   If True, weights are initialized to zero instead of small random
        numbers in the interval \[0,1\]; ignored if
        solver=’normal\_equation’

    [<span id="id2" class="problematic">w\_</span>](#id1) <span class="classifier-delimiter">:</span> <span class="classifier">1d-array</span>
    :   Weights after fitting.

    [<span id="id4" class="problematic">cost\_</span>](#id3) <span class="classifier-delimiter">:</span> <span class="classifier">list</span>
    :   Sum of squared errors after each epoch; ignored if
        solver=’normal\_equation’

     `fit`{.descname}<span class="sig-paren">(</span>*X*, *y*, *init\_weights=True*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.fit)[¶](#mlxtend.regressor.linear_regression.LinearRegression.fit "Permalink to this definition")

    :   Fit training data.

        X <span class="classifier-delimiter">:</span> <span class="classifier">{array-like, sparse matrix}, shape = \[n\_samples, n\_features\]</span>
        :   Training vectors, where n\_samples is the number of samples
            and n\_features is the number of features.

        y <span class="classifier-delimiter">:</span> <span class="classifier">array-like, shape = \[n\_samples\]</span>
        :   Target values.

        init\_weights <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: True)</span>
        :   (Re)initializes weights to small random floats if True.

        self : object

     `net_input`{.descname}<span class="sig-paren">(</span>*X*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.net_input)[¶](#mlxtend.regressor.linear_regression.LinearRegression.net_input "Permalink to this definition")

    :   Net input function

     `predict`{.descname}<span class="sig-paren">(</span>*X*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.predict)[¶](#mlxtend.regressor.linear_regression.LinearRegression.predict "Permalink to this definition")

    :   Predict target values for X.

        X <span class="classifier-delimiter">:</span> <span class="classifier">{array-like, sparse matrix}, shape = \[n\_samples, n\_features\]</span>
        :   Training vectors, where n\_samples is the number of samples
            and n\_features is the number of features.

        float : Predicted target value.

</div>

<div id="module-mlxtend.regressor" class="section">

<span id="module-contents"></span>
Module contents[¶](#module-mlxtend.regressor "Permalink to this headline")
--------------------------------------------------------------------------

 *class* `mlxtend.regressor.`{.descclassname}`LinearRegression`{.descname}<span class="sig-paren">(</span>*solver='normal\_equation'*, *eta=0.01*, *epochs=50*, *random\_seed=None*, *shuffle=False*, *zero\_init\_weight=False*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression)[¶](#mlxtend.regressor.LinearRegression "Permalink to this definition")

:   Bases: `object`{.xref .py .py-class .docutils .literal}

    Ordinary least squares linear regression.

    solver <span class="classifier-delimiter">:</span> <span class="classifier">{‘gd’, ‘sgd’, ‘normal\_equation’} (default: ‘normal\_equation’)</span>
    :   Method for solving the cost function. ‘gd’ for gradient descent,
        ‘sgd’ for stochastic gradient descent, or
        ‘normal\_equation’ (default) to solve the cost
        function analytically.

    eta <span class="classifier-delimiter">:</span> <span class="classifier">float (default: 0.1)</span>
    :   Learning rate (between 0.0 and 1.0); ignored
        if solver=’normal\_equation’.

    epochs <span class="classifier-delimiter">:</span> <span class="classifier">int (default: 50)</span>
    :   Passes over the training dataset; ignored
        if solver=’normal\_equation’.

    shuffle <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: False)</span>
    :   Shuffles training data every epoch if True to prevent circles;
        ignored if solver=’normal\_equation’.

    random\_seed <span class="classifier-delimiter">:</span> <span class="classifier">int (default: None)</span>
    :   Set random state for shuffling and initializing the weights;
        ignored if solver=’normal\_equation’.

    zero\_init\_weight <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: False)</span>
    :   If True, weights are initialized to zero instead of small random
        numbers in the interval \[0,1\]; ignored if
        solver=’normal\_equation’

    [<span id="id6" class="problematic">w\_</span>](#id5) <span class="classifier-delimiter">:</span> <span class="classifier">1d-array</span>
    :   Weights after fitting.

    [<span id="id8" class="problematic">cost\_</span>](#id7) <span class="classifier-delimiter">:</span> <span class="classifier">list</span>
    :   Sum of squared errors after each epoch; ignored if
        solver=’normal\_equation’

     `fit`{.descname}<span class="sig-paren">(</span>*X*, *y*, *init\_weights=True*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.fit)[¶](#mlxtend.regressor.LinearRegression.fit "Permalink to this definition")

    :   Fit training data.

        X <span class="classifier-delimiter">:</span> <span class="classifier">{array-like, sparse matrix}, shape = \[n\_samples, n\_features\]</span>
        :   Training vectors, where n\_samples is the number of samples
            and n\_features is the number of features.

        y <span class="classifier-delimiter">:</span> <span class="classifier">array-like, shape = \[n\_samples\]</span>
        :   Target values.

        init\_weights <span class="classifier-delimiter">:</span> <span class="classifier">bool (default: True)</span>
        :   (Re)initializes weights to small random floats if True.

        self : object

     `net_input`{.descname}<span class="sig-paren">(</span>*X*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.net_input)[¶](#mlxtend.regressor.LinearRegression.net_input "Permalink to this definition")

    :   Net input function

     `predict`{.descname}<span class="sig-paren">(</span>*X*<span class="sig-paren">)</span>[<span class="viewcode-link">\[source\]</span>](_modules/mlxtend/regressor/linear_regression.html#LinearRegression.predict)[¶](#mlxtend.regressor.LinearRegression.predict "Permalink to this definition")

    :   Predict target values for X.

        X <span class="classifier-delimiter">:</span> <span class="classifier">{array-like, sparse matrix}, shape = \[n\_samples, n\_features\]</span>
        :   Training vectors, where n\_samples is the number of samples
            and n\_features is the number of features.

        float : Predicted target value.

</div>

</div>

</div>

</div>

</div>

<div class="sphinxsidebar" role="navigation"
aria-label="main navigation">

<div class="sphinxsidebarwrapper">

### [Table Of Contents](index.html)

-   [mlxtend.regressor package](#)
    -   [Submodules](#submodules)
    -   [mlxtend.regressor.linear\_regression
        module](#module-mlxtend.regressor.linear_regression)
    -   [Module contents](#module-mlxtend.regressor)

<div class="relations">

### Related Topics

-   [Documentation overview](index.html)
    -   [mlxtend package](mlxtend.html)
        -   Previous: [mlxtend.regression\_utils
            package](mlxtend.regression_utils.html "previous chapter")
        -   Next: [mlxtend.text
            package](mlxtend.text.html "next chapter")

</div>

<div role="note" aria-label="source link">

### This Page

-   [Show Source](_sources/mlxtend.regressor.txt)

</div>

<div id="searchbox" style="display: none" role="search">

### Quick search

Enter search terms or a module, class or function name.

</div>

</div>

</div>

<div class="clearer">

</div>

</div>

<div class="footer">

©2015, Author. | Powered by [Sphinx 1.3.1](http://sphinx-doc.org/) &
[Alabaster 0.7.6](https://github.com/bitprophet/alabaster) | [Page
source](_sources/mlxtend.regressor.txt)

</div>
