# Installing mlxtend

---

### PyPI

To install mlxtend, just execute  

```bash
pip install mlxtend  
```

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the following command from inside the mlxtend folder:

```bash
pip install .
```

##### Upgrading via `pip`

To upgrade an existing version of mlxtend from PyPI, execute

```bash
pip install mlxtend --upgrade --no-deps
```

Please note that the dependencies (NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

##### Installing mlxtend from the source distribution

In rare cases, users reported problems on certain systems with the default `pip` installation command, which installs mlxtend from the binary distribution ("wheels") on PyPI. If you should encounter similar problems, you could try to install mlxtend from the source distribution instead via

```bash
pip install --no-binary :all: mlxtend
```

Also, I would appreciate it if you could report any issues that occur when using `pip install mlxtend` in hope that we can fix these in future releases.

### Conda

The mlxtend package is also [available through conda forge](https://github.com/conda-forge/mlxtend-feedstock). 

To install mlxtend using conda, use the following command:

    conda install mlxtend --channel conda-forge

or simply 

    conda install mlxtend

if you added conda-forge to your channels (`conda config --add channels conda-forge`).

### Dev Version

The mlxtend version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/rasbt/mlxtend.git
```

Or, you can fork the GitHub repository from https://github.com/rasbt/mlxtend and install mlxtend from your local drive via

```bash
pip install .
```
