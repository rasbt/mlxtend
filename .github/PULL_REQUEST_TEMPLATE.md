Please make sure that these boxes are checked before submitting a new pull request -- thank you!

- [ ] Make sure you submit this pull request as a separate topic branch (and not to "master")
- [ ] Provide a small summary describing the Pull Request below:

- [ ] Link to the respective issue on the [Issue Tracker](https://github.com/rasbt/mlxtend/issues) if one exists. E.g.,

Fixes #<ISSUE_NUMBER>

- [ ] Add appropriate unit test functions in the `./mlxtend/*/tests` directories
- [ ] Run `nosetests ./mlxtend -sv` and make sure that all unit tests pass
- [ ] Check/improve the test coverage by running `nosetests ./mlxtend --with-coverage`
- [ ] Check for style issues by running `flake8 ./mlxtend` (you may want to run nosetests again after you made modifications to the code)
- [ ] Add a note about the modification or contribution to the `./docs/sources/`CHANGELOG.md` file
- [ ] Modify documentation in the appropriate location under `mlxtend/docs/sources/`
- [ ] Push the topic branch to the server and create a pull request
- [ ] Check the Travis-CI build passed at https://travis-ci.org/rasbt/mlxtend

**Note:**  
**Due to the improved GitHub UI, the squashing of commits is no longer necessary. (Please don't squash commits since they help with keeping track of the changes during the discussion).**

For more information and instructions, please see http://rasbt.github.io/mlxtend/contributing/
