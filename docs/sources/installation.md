# Installing mlxtend

---

### Using uv

To add mlxtend to a uv-managed project, run

```bash
uv add mlxtend
```

For a one-off command without changing your current project, run

```bash
uv run --with mlxtend python -c "import mlxtend; print(mlxtend.__version__)"
```

##### Upgrading via uv

To upgrade an existing uv project dependency, run

```bash
uv add -U mlxtend
```

This updates the dependency in your project and syncs the environment.

##### Installing mlxtend from the source distribution

In rare cases, users reported problems on certain systems with the default binary distribution from PyPI. If you encounter a similar problem, try installing mlxtend from the source distribution instead:

```bash
uv add --no-binary-package mlxtend mlxtend
```

Also, I would appreciate it if you could report any issues that occur when installing mlxtend so we can fix these in future releases.

### Dev Version

The mlxtend version on PyPI may always be one step behind; you can install the latest development version from the GitHub repository by executing

```bash
uv add "mlxtend @ git+https://github.com/rasbt/mlxtend.git"
```

Or, you can fork the GitHub repository from https://github.com/rasbt/mlxtend and run mlxtend from your local checkout via

```bash
git clone https://github.com/<your_username>/mlxtend.git
cd mlxtend
uv sync --group dev
uv run python -c "import mlxtend; print(mlxtend.__version__)"
```
