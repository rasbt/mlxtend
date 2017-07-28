# Installing mlxtend

---

### PyPI

To install mlxtend, just execute  

```bash
pip install mlxtend  
```

Alternatively, you download the package manually from the Python Package Index [https://pypi.python.org/pypi/mlxtend](https://pypi.python.org/pypi/mlxtend), unzip it, navigate into the package, and use the command:

```bash
python setup.py install
```

##### Upgrading via `pip`

To upgrade an existing version of mlxtend from PyPI, execute

```bash
pip install mlxtend --upgrade --no-deps
```

Please note that the dependencies (NumPy and SciPy) will also be upgraded if you omit the `--no-deps` flag; use the `--no-deps` ("no dependencies") flag if you don't want this.

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
python setup.py install
```
