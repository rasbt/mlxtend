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

### Dev Version

The mlxtend version on PyPI may always one step behind; you can install the latest development version from the GitHub repository by executing

```bash
pip install git+git://github.com/rasbt/mlxtend.git#egg=mlxtend
```

Or, you can fork the GitHub repository from https://github.com/rasbt/mlxtend and install mlxtend from your local drive via

```bash
python setup.py install
```

### Anaconda/Conda

Conda packages are now available for Mac, Windows, and Linux. You can install mlxtend using conda by executing 

```bash
conda install -c rasbt mlxtend
```