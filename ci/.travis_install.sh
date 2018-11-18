#!/usr/bin/env bash

set -e

if [[ "$TRAVIS_OS_NAME" != "osx" ]]; then 
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    fi
else
    if [ "${TRAVIS_PYTHON_VERSION}" == "2.7" ]; then

        wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh
    else
        wget https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
    fi
fi
    

bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda update -q pip
conda info -a

conda create -q -n test-environment python=$MINICONDA_PYTHON_VERSION
source activate test-environment

if [ "${LATEST}" = "true" ]; then
    pip install ".[testing]"
else
    conda install numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION pandas=$PANDAS_VERSION scikit-learn=$SKLEARN_VERSION
    pip install ".[testing]"
fi

conda install jupyter

if [ "${IMAGE}" = "true" ]; then
    pip install dlib
    pip install imageio
    pip install scikit-image
fi

if [ "${COVERAGE}" = "true" ]; then
    conda install coveralls
fi

pip install nose-exclude

python --version
python -c "import pandas; print('pandas %s' % pandas.__version__)"
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import sklearn; print('sklearn %s' % sklearn.__version__)"
python -c "import mlxtend; print('mlxtend %s' % mlxtend.__version__)"
