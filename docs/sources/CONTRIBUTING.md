# How to Contribute

---

I would be very happy about any kind of contributions that help to improve and extend the functionality of mlxtend.


## Quick Contributor Checklist

This is a quick checklist about the different steps of a typical contribution to mlxtend (and
other open source projects). Consider copying this list to a local text file (or the issue tracker)
and checking off items as you go.



### 1) Making and testing code changes:

1. [ ]  Open a new "issue" on GitHub to discuss the new feature / bug fix  
2. [ ]  Fork the mlxtend repository from GitHub (if not already done earlier)
3. [ ]  Create and check out a new topic branch (please don't make modifications in the master branch)
4. [ ]  Implement the new feature or apply the bug-fix  
5. [ ]  Add appropriate unit test functions in `mlxtend/*/tests`
6. [ ]  Run `uv run --group dev python -m pytest ./mlxtend -sv` and make sure that all unit tests pass  

7. [ ] Make sure the newly implemented feature has good test coverage:

```
uv sync --group dev
# test all: 
# uv run --group dev python -m coverage run --source=mlxtend --branch -m pytest .
uv run --group dev python -m coverage run --source=mlxtend --branch -m pytest mlxtend/<insert_path>
uv run --group dev python -m coverage html
```


8. [ ]  Modify documentation in the appropriate location under `mlxtend/docs/sources/`  

9. [ ]  Add a note about the modification/contribution to the `./docs/sources/changelog.md` file  



### 2) Checking code style:

When you check in a PR, mlxtend will run code style checks via flak8 and black. To make the contributor experience easier, we recommend you check the code style locally before pushing it to the repository. This way it is less likely that the automated checkers will complain and prompt you to make fixes.

There are two ways you can do this:

**Option A**: Running the tools manually

1. [ ]  Check for style issues by running `uv run --group dev python -m flake8 ./mlxtend` (you may want to run `uv run --group dev python -m pytest` again after you made modifications to the code)
2. [ ]  We recommend using [black](https://black.readthedocs.io/en/stable/) to format the code automatically according to recommended style changes. After running `uv sync --group dev`, you can do this via 

```
uv run --group dev python -m black [source_file_or_directory]
```

3. [ ] Run [`isort`](https://pycqa.github.io/isort/) which will sort the imports alphabetically. We recommend the following command:

```
uv run --group dev python -m isort -p mlxtend --line-length 88 --multi-line 3 --profile black mypythonfile.py
```

**Option B**: Using pre-commit hooks (recommended)

The pre-commit hooks for mlxtend will check your code via `flake8`, `black`, and `isort` automatically before you make a `git commit`. You can read more about pre-commit hooks [here](https://dev.to/m1yag1/how-to-setup-your-project-with-pre-commit-black-and-flake8-183k).

1. [ ] Install the development dependencies via `uv sync --group dev`.
2. [ ] In the `mlxtend` folder, run `uv run --group dev python -m pre_commit install` (you only have to do it once).





### 3) Submitting your code

1. [ ]  Push the topic branch to the server and create a pull request.
2. [ ]  Check the automated tests passed.
3. [ ] The automatic [PEP8](https://peps.python.org/pep-0008/)/[black](https://black.readthedocs.io/en/stable/) integrations may prompt you to modify the code stylistically. It would be nice if  you could apply the suggested changes.



<hr>

# Tips for Contributors


## Getting Started - Creating a New Issue and Forking the Repository

- If you don't have a [GitHub](https://github.com) account, yet, please create one to contribute to this project.
- Please submit a ticket for your issue to discuss the fix or new feature before too much time and effort is spent for the implementation.

![](./img/contributing/new_issue.png)

- Fork the `mlxtend` repository from the GitHub web interface.

![](./img/contributing/fork.png)

- Clone the `mlxtend` repository to your local machine by executing
 ```git clone https://github.com/<your_username>/mlxtend.git```

## Syncing an Existing Fork

If you already forked mlxtend earlier, you can bring you "Fork" up to date
with the master branch as follows:

#### 1. Configuring a remote that points to the upstream repository on GitHub

List the current configured remote repository of your fork by executing

```bash
$ git remote -v
```

If you see something like

```bash
origin	https://github.com/<your username>/mlxtend.git (fetch)
origin	https://github.com/<your username>/mlxtend.git (push)
```
you need to specify a new remote *upstream* repository via

```bash
$ git remote add upstream https://github.com/rasbt/mlxtend.git
```

Now, verify the new upstream repository you've specified for your fork by executing

```bash
$ git remote -v
```

You should see following output if everything is configured correctly:

```bash
origin	https://github.com/<your username>/mlxtend.git (fetch)
origin	https://github.com/<your username>/mlxtend.git (push)
upstream	https://github.com/rasbt/mlxtend.git (fetch)
upstream	https://github.com/rasbt/mlxtend.git (push)
```

#### 2. Syncing your Fork

First, fetch the updates of the original project's master branch by executing:

```bash
$ git fetch upstream
```

You should see the following output

```bash
remote: Counting objects: xx, done.
remote: Compressing objects: 100% (xx/xx), done.
remote: Total xx (delta xx), reused xx (delta x)
Unpacking objects: 100% (xx/xx), done.
From https://github.com/rasbt/mlxtend
 * [new branch]      master     -> upstream/master
```

This means that the commits to the `rasbt/mlxtend` master branch are now
stored in the local branch `upstream/master`.

If you are not already on your local project's master branch, execute

```bash
$ git checkout master
```

Finally, merge the changes in upstream/master to your local master branch by
executing

```bash
$ git merge upstream/master
```

which will give you an output that looks similar to

```bash
Updating xxx...xxx
Fast-forward
SOME FILE1                    |    12 +++++++
SOME FILE2                    |    10 +++++++
2 files changed, 22 insertions(+),
```


## *The Main Workflow - Making Changes in a New Topic Branch

Listed below are the 9 typical steps of a contribution.

#### 1. Discussing the Feature or Modification

Before you start coding, please discuss the new feature, bugfix, or other modification to the project
on the project's [issue tracker](https://github.com/rasbt/mlxtend/issues). Before you open a "new issue," please
do a quick search to see if a similar issue has been submitted already.

#### 2. Creating a new feature branch

Please avoid working directly on the master branch but create a new feature branch:

```bash
$ git branch <new_feature>
```

Switch to the new feature branch by executing

```bash
$ git checkout <new_feature>
```

#### 3. Developing the new feature / bug fix

Now it's time to modify existing code or to contribute new code to the project.

#### 4. Testing your code

Add the respective unit tests and check if they pass:

```bash
$ PYTHONPATH='.' pytest ./mlxtend ---with-coverage
```


#### 5. Documenting changes

Please add an entry to the `mlxtend/docs/sources/changelog.md` file.
If it is a new feature, it would also be nice if you could update the documentation in appropriate location in `mlxtend/sources`.


#### 6. Committing changes

When you are ready to commit the changes, please provide a meaningful `commit` message:

```bash
$ git add <modifies_files> # or `git add .`
$ git commit -m '<meaningful commit message>'
```

#### 7. Optional: squashing commits

If you made multiple smaller commits, it would be nice if you could group them into a larger, summarizing commit. First, list your recent commit via

**Note**  
**Due to the improved GitHub UI, this is no longer necessary/encouraged.**


```bash
$ git log
```

which will list the commits from newest to oldest in the following format by default:


```bash
commit 046e3af8a9127df8eac879454f029937c8a31c41
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 03:46:37 2015 -0500

    fixed setup.py

commit c3c00f6ba0e8f48bbe1c9081b8ae3817e57ecc5c
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 03:04:39 2015 -0500

        documented feature x

commit d87934fe8726c46f0b166d6290a3bf38915d6e75
Author: rasbt <mail@sebastianraschka.com>
Date:   Tue Nov 24 02:44:45 2015 -0500

        added support for feature x
```

Assuming that it would make sense to group these 3 commits into one, we can execute

```bash
$ git rebase -i HEAD~3
```

which will bring our default git editor with the following contents:

```bash
pick d87934f added support for feature x
pick c3c00f6 documented feature x
pick 046e3af fixed setup.py
```

Since `c3c00f6` and `046e3af` are related to the original commit of `feature x`, let's keep the `d87934f` and squash the 2 following commits into this initial one by changes the lines to


```
pick d87934f added support for feature x
squash c3c00f6 documented feature x
squash 046e3af fixed setup.py
```

Now, save the changes in your editor. Now, quitting the editor will apply the `rebase` changes, and the editor will open a second time, prompting you to enter a new commit message. In this case, we could enter `support for feature x` to summarize the contributions.


#### 8. Uploading changes

Push your changes to a topic branch to the git server by executing:

```bash
$ git push origin <feature_branch>
```

#### 9. Submitting a `pull request`

Go to your GitHub repository online, select the new feature branch, and submit a new pull request:


![](./img/contributing/pull_request.png)


<hr>

# Notes for Developers



## Building the documentation

The documentation is built via [MkDocs](https://www.mkdocs.org); to ensure that the documentation is rendered correctly, you can view the documentation locally by executing `mkdocs serve` from the `mlxtend/docs` directory.

For example,

```bash
~/github/mlxtend/docs$ mkdocs serve
```

### 1. Building the API documentation

To build the API documentation, navigate to `mlxtend/docs` and execute the `make_api.py` file from this directory via

```python
~/github/mlxtend/docs$ python make_api.py
```

This should place the API documentation into the correct directories into the two directories:

- `mlxtend/docs/sources/api_modules`
- `mlxtend/docs/sources/api_subpackes`

### 2. Editing the User Guide

The documents containing code examples for the "User Guide" are generated from IPython Notebook files. In order to convert a IPython notebook file to markdown after editing, please follow the following steps:

1. Modify or edit the existing notebook.
2. Execute all cells in the current notebook and make sure that no errors occur.
3. Convert the notebook to markdown using the `ipynb2markdown.py` converter

```python
~/github/mlxtend/docs$ python ipynb2markdown.py --ipynb ./sources/user_guide/subpackage/notebookname.ipynb
```

**Note**  

If you are adding a new document, please also include it in the pages section in the `mlxtend/docs/mkdocs.yml` file.



### 3. Building static HTML files of the documentation

First, please check the documenation via localhost (https://127.0.0.1:8000/):

```bash
~/github/mlxtend/docs$ mkdocs serve
```

Next, build the static HTML files of the mlxtend documentation via

```bash
~/github/mlxtend/docs$ mkdocs build --clean
```

To deploy the documentation, execute

```bash
~/github/mlxtend/docs$ mkdocs gh-deploy --clean
```

### 4. Generate a PDF of the documentation

To generate a PDF version of the documentation, simply `cd` into the `mlxtend/docs` directory and execute:

```bash
python md2pdf.py
```

## Uploading a new version to PyPI

### 1. Creating a new testing environment

From the mlxtend repository root, create the uv environment with the development tools:

```bash
$ uv sync --group dev --python 3.11
```

### 2. Installing the package from local files

Test the editable local checkout:

```bash
$ uv run python -c 'import mlxtend; print(mlxtend.__file__)'
```

Run the test suite from the same uv environment:

```bash
$ uv run --group dev python -m pytest mlxtend
```

### 3. Building and checking the package

Consider deploying the package to the PyPI test server first. The setup instructions can be found [here](https://wiki.python.org/moin/TestPyPI).

Create the source distribution and wheel in the `./dist` directory:

```bash
$ uv build
```

Check the distribution metadata:

```bash
$ uv run --group dev python -m twine check --strict dist/*
```

Install the wheel and source distribution in isolated uv environments to make sure both work.
The distribution filenames change with each version.

```bash
$ uv run --isolated --with ./dist/mlxtend-*.tar.gz python -c 'import mlxtend; print(mlxtend.__version__)'
$ uv run --isolated --with ./dist/mlxtend-*-py3-none-any.whl python -c 'import mlxtend; print(mlxtend.__version__)'
```

### 4. Deploying the package

Upload the package to the PyPI test server:

```bash
$ uv run --group dev python -m twine upload --repository testpypi dist/*
```

Then, install it from the PyPI test server and check that it imports:

```bash
$ uv run --isolated --default-index https://test.pypi.org/simple/ --index https://pypi.org/simple --with mlxtend python -c 'import mlxtend; print(mlxtend.__version__)'
```

After this dry run succeeds, repeat the upload using the real PyPI repository:

```bash
$ uv run --group dev python -m twine upload dist/*
```

**Note**:

if you get an error like

    HTTPError: 403 Forbidden from https://upload.pypi.org/legacy/

make sure you have an up to date version of twine in the uv development environment.
