# How to Contribute

I would be very happy about any kind of contributions that help to improve and extend the functionality of mlxtend.

<br>
<br>

### Quick contributor checklist

[ ]  Open a new "issue" on GitHub to discuss the new feature / bugfix  
[ ]  Create and checkout a new topic branch   
[ ]  Implement new feature or apply bugfix  
[ ]  Add appropriate unit test functions  
[ ]  Run `nosetests -sv` and make sure that all unit tests pass  
[ ]  Add a note about the change to the `./docs/sources/CHANGELOG.md` file  
[ ]  Modify documentation in `./docs/sources/` if appropriate  
[ ]  Push the topic branch to the server and create a pull request

<br>


## Getting Started

- If you don't have a [GitHub](https://github.com) account yet, please create one to contribute to this project.
- Please submit a ticket for your issue to discuss the fix or new feature before too much time and effort is spent for the implementation.

![](img/contributing/new_issue.png)

- Fork the `mlxtend` repository from the GitHub web interface.

![](img/contributing/fork.png)

- Clone the `mlxtend` repository to your local machine
	- `git clone https://github.com/your_username/mlxtend.git`

<br>
<br>

## Making Changes

- Please avoid working directly on the master branch but create a new feature branch:
	- `git branch new_feature`
	- `git checkout new_feature`
- When you make changes, please provide meaningful `commit` messages:
	- `git add edited_file` 
	- `git commit -m 'my note'` 
- Make an entry in the `mlxtend/docs/source/changelog.md` file.
- Add tests to the `mlxtend/tests` directory.
- Run all tests (e.g., via `nosetests`  or `pytest`).
- If it is a new feature, it would be nice (but not necessary) if you could update the documentation.
- Push your changes to a topic branch:
	- `git push -u origin my_feature`
- Submit a `pull request` from your forked repository via the GitHub web interface.
![](img/contributing/pull_request.png)

<br>
<br>

## Notes for Developers

### Building the documentation

The documentation is build using [mkdocs](http://www.mkdocs.org). 
If you are adding a new document, please add it to the `~/github/mlxtend/docs/mkdocs.yml` file.

In order to view the documentation locally, execute `mkdocs serve` from the `mlxtend/docs` directory.

For example,
	
	~/github/mlxtend/docs >mkdocs serve

	
If everything looks fine, the documentation can be build by executing

	~/github/mlxtend/docs >mkdocs build
	
<br>
	
