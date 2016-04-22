#!/usr/bin/env bash

set -e

if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then
    wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh;
else
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
fi

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a

if [ "${LATEST}" = "true" ]; then
    conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy scipy matplotlib pandas scikit-learn;
else
    conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION matplotlib=$MATPLOTLIB_VERSION pandas=$PANDAS_VERSION scikit-learn=$SKLEARN_VERSION;
fi

source activate test-environment

pip install nose nose-exclude;

if [ "${COVERAGE}" = "true" ]; then
    pip install coverage coveralls;
fi

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)";
python -c "import matplotlib; print('matplotlib %s' % matplotlib.__version__)";
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"

if [ "${TENSORFLOW}" = "true" ]; then
    if [ "${TRAVIS_PYTHON_VERSION}" = "2.7" ]; then
        pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp27-none-linux_x86_64.whl;
    else
        pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.8.0-cp34-cp34m-linux_x86_64.whl;
    fi
    python -c "import tensorflow; print('tensorflow %s' % tensorflow.__version__)";
else
    python -c "import sklearn; print('sklearn %s' % sklearn.__version__)";
fi

python setup.py install;
