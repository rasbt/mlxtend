### Code of Conduct

<!-- 
If this is your first Pull Request for the MLxtend repository, please review
the code of conduct, which is available at https://rasbt.github.io/mlxtend/Code-of-Conduct/. 
-->


### Description

<!--  
Please insert a brief description of the Pull request below.
-->

### Related issues or pull requests

<!--  
If applicable, please link related issues/pull request below. For example,   
"Fixes #366". Note that the "Fixes" keyword in GitHub will automatically
close the listed issue upon merging this Pull Request.
-->

### Pull Request Checklist

<!--
Please fill out the following checklist if applicable. For more more information and help, please see the Contributor Documentation avaialable at https://rasbt.github.io/mlxtend/contributing/.
-->

- [ ] Added a note about the modification or contribution to the `./docs/sources/CHANGELOG.md` file (if applicable)
- [ ] Added appropriate unit test functions in the `./mlxtend/*/tests` directories (if applicable)
- [ ] Modify documentation in the corresponding Jupyter Notebook under `mlxtend/docs/sources/` (if applicable)
- [ ] Ran `PYTHONPATH='.' pytest ./mlxtend -sv` and make sure that all unit tests pass (for small modifications, it might be sufficient to only run the specific test file, e.g., `PYTHONPATH='.' pytest ./mlxtend/classifier/tests/test_stacking_cv_classifier.py -sv`)
- [ ] Checked for style issues by running `flake8 ./mlxtend`


<!--NOTE  
Due to the improved GitHub UI, the squashing of commits is no longer necessary.
Please DO NOT SQUASH commits since they help with keeping track of the changes during the discussion).
-->
